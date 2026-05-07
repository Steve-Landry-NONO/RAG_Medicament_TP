# -*- coding: utf-8 -*-
"""
app.py

Interface Streamlit pour le TP RAG - Assistant Médicaments.
Déployable sur Streamlit Community Cloud via GitHub.

La clé GROQ_API_KEY est lue depuis :
  - st.secrets       → Streamlit Cloud (Settings > Secrets)
  - .env             → développement local
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag import (
    AVERTISSEMENT_MEDICAL,
    SCORE_MIN_ACCEPTABLE,
    charger_index_et_chunks,
    charger_modele_embedding,
    charger_prompt_systeme,
    generer_reponse,
    rechercher,
)


# ============================================================
# Configuration de la page (doit être le premier appel Streamlit)
# ============================================================

st.set_page_config(
    page_title="Assistant Médicaments — TP RAG",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# Chargement des ressources lourdes (mis en cache côté serveur)
# ============================================================

@st.cache_resource(show_spinner="Chargement de la base de connaissances…")
def charger_ressources():
    """
    Charge et met en cache l'index FAISS, les chunks, le modèle d'embedding
    et le prompt système.
    @st.cache_resource garantit un seul chargement par redémarrage serveur,
    peu importe le nombre d'utilisateurs simultanés.
    """
    index, chunks_avec_meta = charger_index_et_chunks()
    modele = charger_modele_embedding()
    prompt_systeme = charger_prompt_systeme()
    return index, chunks_avec_meta, modele, prompt_systeme


def get_groq_client() -> Groq:
    """
    Initialise le client Groq.
    Priorité : st.secrets (Streamlit Cloud) > fichier .env (local).
    """
    api_key = None

    # Streamlit Cloud injecte les secrets via st.secrets
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

    # Fallback : fichier .env pour le développement local
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error(
            "**Clé GROQ_API_KEY introuvable.**\n\n"
            "- **Local** : crée un fichier `.env` avec `GROQ_API_KEY=ta_clé`\n"
            "- **Streamlit Cloud** : ajoute la clé dans *Settings → Secrets*"
        )
        st.stop()

    return Groq(api_key=api_key)


# ============================================================
# Affichage des sources dans un expander
# ============================================================

def afficher_sources_streamlit(chunks: list[dict]) -> None:
    """
    Affiche les sources récupérées dans un expander Streamlit.
    """
    if not chunks:
        return

    with st.expander("📚 Sources utilisées"):
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            score = chunk.get("score", 0.0)
            medicament = meta.get("medicament", "Médicament inconnu")
            section = meta.get("section", "section inconnue")
            code_cis = meta.get("code_cis", "?")
            contenu = chunk.get("contenu", "")

            st.markdown(
                f"**[{i}] {medicament}** · "
                f"section : *{section}* · "
                f"CIS `{code_cis}` · "
                f"score `{score:.3f}`"
            )
            st.caption(contenu[:400] + ("…" if len(contenu) > 400 else ""))

            if i < len(chunks):
                st.divider()


# ============================================================
# Interface principale
# ============================================================

def main() -> None:

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/fr/thumb/4/48/Ansm.svg/200px-Ansm.svg.png",
            width=120,
        )
        st.markdown("### 💊 Assistant Médicaments")
        st.markdown(
            "Basé sur les notices officielles **ANSM** (base CIS_RCP).\n\n"
            "**Médicaments disponibles :**\n"
            "Doliprane, Dafalgan, Efferalgan, Advil, Nurofen, "
            "Aspirine, Aspégic, Augmentin, Amoxicilline, "
            "Smecta, Imodium, Ventoline, Bécotide, "
            "Oméprazole, Inexium, Metformine, Glucophage"
        )
        st.divider()
        st.markdown(
            "**Exemples de questions :**\n"
            "- Quelle est la posologie du Doliprane adulte ?\n"
            "- L'ibuprofène est-il contre-indiqué chez la femme enceinte ?\n"
            "- Quels sont les effets indésirables de l'amoxicilline ?"
        )
        st.divider()
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("TP RAG réalisé par:  Steve KOUOKAM & Ludovic TUEKAM")

    # ── En-tête ────────────────────────────────────────────
    st.title("💊 Assistant Médicaments")
    st.warning(
        "**Avertissement médical** : cet assistant est un outil pédagogique. "
        "Il ne remplace pas l'avis d'un médecin ou d'un pharmacien.",
        icon="⚠️",
    )

    # ── Chargement des ressources ──────────────────────────
    try:
        index, chunks_avec_meta, modele, prompt_systeme = charger_ressources()
        client = get_groq_client()
    except FileNotFoundError as e:
        st.error(f"**Base de connaissances introuvable.**\n\n{e}")
        st.info("Lance `python indexation.py` pour générer l'index FAISS.")
        st.stop()
    except Exception as e:
        st.error(f"**Erreur au démarrage :** {e}")
        st.stop()

    # ── Historique de conversation ─────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Message d'accueil si conversation vide
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Bonjour ! Je suis votre assistant d'information sur les médicaments. "
                "Posez-moi une question sur un médicament disponible dans ma base "
                "(posologie, effets indésirables, contre-indications, interactions…)."
            )

    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                afficher_sources_streamlit(message["sources"])

    # ── Zone de saisie ─────────────────────────────────────
    if question := st.chat_input("Posez votre question sur un médicament…"):

        # Affichage de la question utilisateur
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Génération de la réponse
        with st.chat_message("assistant"):
            chunks_pertinents = []

            with st.spinner("Recherche dans la base de connaissances…"):
                try:
                    chunks_pertinents = rechercher(
                        question=question,
                        modele=modele,
                        index=index,
                        chunks_avec_meta=chunks_avec_meta,
                    )
                    reponse = generer_reponse(
                        question=question,
                        chunks_pertinents=chunks_pertinents,
                        client=client,
                        prompt_systeme=prompt_systeme,
                    )
                except RuntimeError as e:
                    # Erreurs Groq remontées avec message lisible
                    reponse = f"{e}"
                    chunks_pertinents = []
                except Exception as e:
                    reponse = f"Erreur inattendue : {e}"
                    chunks_pertinents = []

            st.markdown(reponse)
            afficher_sources_streamlit(chunks_pertinents)

        # Sauvegarde dans l'historique de session
        st.session_state.messages.append({
            "role": "assistant",
            "content": reponse,
            "sources": chunks_pertinents,
        })


if __name__ == "__main__":
    main()
