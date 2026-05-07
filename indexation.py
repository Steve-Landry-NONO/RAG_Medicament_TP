# -*- coding: utf-8 -*-
"""
indexation.py

Script d'indexation pour le TP RAG - Assistant Médicaments.

Objectif :
- Charger le fichier data/CIS_RCP_export.xlsx
- Nettoyer et filtrer les médicaments utiles
- Transformer les notices en documents textuels
- Découper les longs textes en chunks
- Calculer les embeddings avec sentence-transformers
- Créer un index FAISS persistant
- Sauvegarder les chunks + métadonnées dans storage/chunks.json

Contraintes TP :
- Pas de LangChain
- Pas de LlamaIndex
- FAISS persistant
- Métadonnées exploitables pour citer les sources
"""

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Configuration générale
# ============================================================

DATA_PATH = Path("data/CIS_RCP_export.xlsx")
STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "index.faiss"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Médicaments suggérés par le sujet du TP.
# On filtre dessus pour avoir un corpus raisonnable, rapide à indexer,
# et facile à expliquer.
MEDICAMENTS_CIBLES = [
    "doliprane",
    "dafalgan",
    "efferalgan",
    "ibuprofene",
    "ibuprofène",
    "advil",
    "nurofen",
    "aspirine",
    "aspirin",
    "aspegic",
    "aspégic",
    "amoxicilline",
    "augmentin",
    "smecta",
    "imodium",
    "ventoline",
    "becotide",
    "oméprazole",
    "omeprazole",
    "inexium",
    "metformine",
    "glucophage",
]

# Sections médicales utiles pour le RAG.
# Chaque section deviendra un ou plusieurs chunks.
SECTIONS_MEDICALES = [
    "composition",
    "forme_pharmaceutique",
    "indications",
    "posologie",
    "contre_indications",
    "mises_en_garde",
    "interactions",
    "grossesse_allaitement",
    "effets_indesirables",
    "surdosage",
    "excipients",
    "conditions_prescription",
]


# ============================================================
# 2. Fonctions utilitaires de nettoyage
# ============================================================

def normaliser_texte_pour_recherche(texte: Any) -> str:
    """
    Normalise un texte pour faire des comparaisons simples :
    - minuscules
    - suppression approximative de quelques accents fréquents
    - texte vide si NaN
    """
    if pd.isna(texte):
        return ""

    texte = str(texte).lower()

    remplacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ä": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
    }

    for accent, simple in remplacements.items():
        texte = texte.replace(accent, simple)

    return texte


def nettoyer_texte(texte: Any) -> str:
    """
    Nettoie les textes issus du fichier Excel :
    - convertit les NaN en chaîne vide
    - remplace les séparateurs artificiels '|' par des retours à la ligne
    - supprime les espaces multiples
    - garde un texte lisible pour le LLM
    """
    if pd.isna(texte):
        return ""

    texte = str(texte)
    texte = texte.replace("|", "\n")
    texte = texte.replace("\r", "\n")

    # Normalisation des espaces
    texte = re.sub(r"[ \t]+", " ", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)

    return texte.strip()


def valeur_metadata(valeur: Any) -> str:
    """
    Convertit une valeur de métadonnée en string JSON-compatible.
    """
    if pd.isna(valeur):
        return ""
    return str(valeur).strip()


# ============================================================
# 3. Chargement et filtrage du corpus
# ============================================================

def charger_donnees(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Charge le fichier Excel contenant les RCP médicaments.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Place CIS_RCP_export.xlsx dans le dossier data/."
        )

    print(f"[INFO] Chargement du fichier : {path}")
    df = pd.read_excel(path)

    print(f"[INFO] Nombre de lignes brutes : {len(df)}")
    print(f"[INFO] Colonnes : {list(df.columns)}")

    return df


def filtrer_medicaments_cibles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garde uniquement les lignes qui correspondent aux médicaments cibles.
    On filtre principalement sur la dénomination.
    """
    df = df.copy()

    # On retire les lignes sans dénomination, car elles sont inutilisables
    # pour citer correctement les sources.
    df = df[df["denomination"].notna()]
    df = df[df["denomination"].astype(str).str.strip() != ""]

    print(f"[INFO] Lignes avec dénomination : {len(df)}")

    denominations_normalisees = df["denomination"].apply(normaliser_texte_pour_recherche)

    medicaments_normalises = [
        normaliser_texte_pour_recherche(med) for med in MEDICAMENTS_CIBLES
    ]

    # word-boundary (\b) pour éviter les faux positifs sur des sous-chaînes
    # (ex : "aspirin" ne doit pas matcher "salaspirin" ou un mot composite).
    masque = denominations_normalisees.apply(
        lambda denomination: any(
            re.search(r"\b" + re.escape(med) + r"\b", denomination) is not None
            for med in medicaments_normalises
        )
    )

    df_filtre = df[masque].copy()

    print(f"[INFO] Lignes après filtrage médicaments cibles : {len(df_filtre)}")

    if df_filtre.empty:
        raise ValueError(
            "Aucun médicament cible trouvé. Vérifie les noms ou le fichier source."
        )

    return df_filtre


# ============================================================
# 4. Chunking
# ============================================================

def chunker(texte: str, taille_max: int = 900, overlap: int = 120) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.

    Choix pour ce TP :
    - 900 caractères : assez court pour retrouver précisément une section,
      mais assez long pour garder du contexte médical.
    - 120 caractères d'overlap : permet de ne pas perdre une phrase coupée
      entre deux chunks.

    Le chunker essaie d'abord de découper par paragraphes.
    """
    texte = nettoyer_texte(texte)

    if not texte:
        return []

    if len(texte) <= taille_max:
        return [texte]

    paragraphes = [p.strip() for p in texte.split("\n") if p.strip()]

    chunks = []
    chunk_actuel = ""

    for paragraphe in paragraphes:
        # Si un paragraphe seul est très long, on le découpe brutalement.
        if len(paragraphe) > taille_max:
            if chunk_actuel:
                chunks.append(chunk_actuel.strip())
                chunk_actuel = ""

            debut = 0
            while debut < len(paragraphe):
                fin = debut + taille_max
                morceau = paragraphe[debut:fin].strip()
                if morceau:
                    chunks.append(morceau)
                debut = max(fin - overlap, debut + 1)

            continue

        # Si on peut ajouter le paragraphe au chunk courant
        candidat = (chunk_actuel + "\n" + paragraphe).strip()

        if len(candidat) <= taille_max:
            chunk_actuel = candidat
        else:
            if chunk_actuel:
                chunks.append(chunk_actuel.strip())

            # On ajoute un petit chevauchement à partir du chunk précédent.
            if chunks and overlap > 0:
                prefixe = chunks[-1][-overlap:]
                chunk_actuel = (prefixe + "\n" + paragraphe).strip()
            else:
                chunk_actuel = paragraphe

    if chunk_actuel:
        chunks.append(chunk_actuel.strip())

    # Sécurité : retirer les chunks trop petits ou vides
    chunks = [c for c in chunks if len(c.strip()) >= 30]

    return chunks


# ============================================================
# 5. Transformation du DataFrame en chunks avec métadonnées
# ============================================================

def construire_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Transforme chaque ligne médicament en plusieurs chunks :
    un chunk ou plusieurs chunks par section médicale.
    """
    chunks_avec_meta = []

    for _, row in df.iterrows():
        code_cis = valeur_metadata(row.get("code_cis"))
        denomination = valeur_metadata(row.get("denomination"))
        date_mise_a_jour = valeur_metadata(row.get("date_mise_a_jour"))
        titulaire_amm = valeur_metadata(row.get("titulaire_amm"))
        numero_amm = valeur_metadata(row.get("numero_amm"))

        for section in SECTIONS_MEDICALES:
            contenu_section = nettoyer_texte(row.get(section, ""))

            # On ignore les sections vides ou très pauvres.
            if not contenu_section or len(contenu_section) < 30:
                continue

            # On chunke uniquement le contenu brut (sans le préfixe) pour que
            # l'embedding de chaque morceau soit centré sur son contenu médical.
            # taille_max=800 laisse ~100 chars pour le préfixe ci-dessous,
            # soit un chunk total d'environ 900 chars.
            morceaux_bruts = chunker(contenu_section, taille_max=800, overlap=120)

            # Le préfixe médicament/section est ajouté sur chaque morceau
            # pour ancrer sémantiquement tous les chunks, y compris les suivants.
            prefixe = (
                f"Médicament : {denomination}\n"
                f"Code CIS : {code_cis}\n"
                f"Section : {section}\n\n"
            )
            morceaux = [prefixe + morceau for morceau in morceaux_bruts]

            for i, morceau in enumerate(morceaux):
                chunk_id = f"{code_cis}_{section}_{i}"

                chunks_avec_meta.append(
                    {
                        "id": chunk_id,
                        "contenu": morceau,
                        "metadata": {
                            "code_cis": code_cis,
                            "medicament": denomination,
                            "section": section,
                            "source": "CIS_RCP_export.xlsx",
                            "date_mise_a_jour": date_mise_a_jour,
                            "titulaire_amm": titulaire_amm,
                            "numero_amm": numero_amm,
                        },
                    }
                )

    print(f"[INFO] Nombre total de chunks créés : {len(chunks_avec_meta)}")

    if not chunks_avec_meta:
        raise ValueError("Aucun chunk créé. Vérifie les données ou les sections.")

    return chunks_avec_meta


# ============================================================
# 6. Embeddings
# ============================================================

def embedder_chunks(chunks_avec_meta: list[dict], modele: SentenceTransformer) -> np.ndarray:
    """
    Transforme les chunks en vecteurs.
    Les vecteurs sont normalisés pour utiliser la similarité cosinus
    avec FAISS IndexFlatIP.
    """
    textes = [chunk["contenu"] for chunk in chunks_avec_meta]

    print(f"[INFO] Calcul des embeddings avec : {EMBEDDING_MODEL_NAME}")
    print(f"[INFO] Nombre de textes à encoder : {len(textes)}")

    vecteurs = modele.encode(
        textes,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vecteurs = np.asarray(vecteurs, dtype=np.float32)

    print(f"[INFO] Shape embeddings : {vecteurs.shape}")

    return vecteurs


# ============================================================
# 7. FAISS : création et sauvegarde
# ============================================================

def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    """
    Crée un index FAISS.

    Comme les embeddings sont normalisés, IndexFlatIP revient à faire
    une recherche par similarité cosinus.
    Plus le score est élevé, plus le résultat est pertinent.
    """
    if vecteurs.ndim != 2:
        raise ValueError("Les vecteurs doivent avoir la forme (n_chunks, dimension).")

    dimension = vecteurs.shape[1]

    print(f"[INFO] Création de l'index FAISS, dimension={dimension}")

    index = faiss.IndexFlatIP(dimension)
    index.add(vecteurs)

    print(f"[INFO] Nombre de vecteurs dans FAISS : {index.ntotal}")

    return index


def sauvegarder_index(index: faiss.Index, chunks_avec_meta: list[dict]) -> None:
    """
    Sauvegarde :
    - l'index FAISS dans storage/index.faiss
    - les chunks + métadonnées dans storage/chunks.json

    Important :
    FAISS retourne des indices numériques.
    Il faut donc conserver chunks.json dans le même ordre que les vecteurs.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Index sauvegardé : {INDEX_PATH}")
    print(f"[OK] Chunks sauvegardés : {CHUNKS_PATH}")


def verifier_sauvegarde() -> None:
    """
    Vérifie rapidement que les fichiers ont bien été créés.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index manquant : {INDEX_PATH}")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks manquants : {CHUNKS_PATH}")

    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[CHECK] Index rechargé : {index.ntotal} vecteurs")
    print(f"[CHECK] Chunks rechargés : {len(chunks)} chunks")

    if index.ntotal != len(chunks):
        raise ValueError(
            "Incohérence : le nombre de vecteurs FAISS ne correspond pas "
            "au nombre de chunks sauvegardés."
        )

    print("[OK] Vérification terminée : index et chunks cohérents.")


# ============================================================
# 8. Programme principal
# ============================================================

def main() -> None:
    """
    Point d'entrée du script.
    """
    print("=" * 70)
    print("INDEXATION RAG - ASSISTANT MÉDICAMENTS")
    print("=" * 70)

    df = charger_donnees(DATA_PATH)
    df_filtre = filtrer_medicaments_cibles(df)

    print("\n[INFO] Exemples de médicaments conservés :")
    exemples = df_filtre["denomination"].dropna().astype(str).head(10).tolist()
    for med in exemples:
        print(f"  - {med}")

    chunks_avec_meta = construire_chunks(df_filtre)

    modele = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vecteurs = embedder_chunks(chunks_avec_meta, modele)

    index = creer_index_faiss(vecteurs)
    sauvegarder_index(index, chunks_avec_meta)
    verifier_sauvegarde()

    print("=" * 70)
    print("[OK] Indexation terminée avec succès.")
    print("=" * 70)


if __name__ == "__main__":
    main()
