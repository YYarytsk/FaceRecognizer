from __future__ import annotations

import streamlit as st

from app_modes import (
    render_chat_mode,
    render_edi_mode,
    render_image_mode,
    render_realtime_mode,
    render_train_mode,
)
from app_shared import LLM_DEVICE, load_face_database


def _initialize_face_database() -> None:
    if "known_embeddings" not in st.session_state or "known_names" not in st.session_state:
        embeddings, names = load_face_database()
        st.session_state.known_embeddings = embeddings
        st.session_state.known_names = names
    embeddings = st.session_state.get("known_embeddings", [])
    names = st.session_state.get("known_names", [])
    if embeddings:
        st.sidebar.success(f"Loaded {len(names)} known face embeddings from disk.")
    else:
        st.sidebar.warning("No saved face database found. Please run 'Train Face Database' first.")


def main() -> None:
    st.title("🔐 Employee Face Recognition & AI Assistant")
    st.sidebar.info(f"LLM running on {LLM_DEVICE.upper()}")
    _initialize_face_database()

    mode = st.sidebar.selectbox(
        "Select Mode",
        [
            "Train Face Database",
            "Real-Time Recognition",
            "Image Recognition",
            "837 File Analyzer",
            "Chat with Assistant",
        ],
    )

    renderers = {
        "Train Face Database": render_train_mode,
        "Real-Time Recognition": render_realtime_mode,
        "Image Recognition": render_image_mode,
        "837 File Analyzer": render_edi_mode,
        "Chat with Assistant": render_chat_mode,
    }
    renderers[mode]()


if __name__ == "__main__":
    main()
