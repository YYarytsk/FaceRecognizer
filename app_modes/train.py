from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from app_shared import EMPLOYEES_DIR, mtcnn, resnet, save_face_database

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _iter_employee_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [path for path in folder.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS]


def render() -> None:
    st.header("📁 Train Face Database")
    st.write("Process all images in the `employees/` folder to encode known faces.")
    st.write("Ensure each filename is the person's name and contains a single face.")

    if st.button("Train/Update Now"):
        image_paths = _iter_employee_images(EMPLOYEES_DIR)
        if not image_paths:
            st.error("No images found in 'employees/' folder. Please add some face images first.")
            return

        embeddings: list[np.ndarray] = []
        names: list[str] = []
        progress = st.progress(0)

        for idx, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as exc:
                st.warning(f"Could not open {img_path.name}: {exc}")
                continue

            face_tensor = mtcnn(image)
            if face_tensor is None:
                st.warning(f"No face detected in {img_path.name}. Skipping.")
                continue

            embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            names.append(img_path.stem)
            progress.progress((idx + 1) / len(image_paths))

        if not embeddings:
            st.error("No embeddings were created. Ensure the images contain clear faces.")
            return

        save_face_database(embeddings, names)
        st.session_state.known_embeddings = embeddings
        st.session_state.known_names = names
        st.success(f"Training completed. Processed {len(names)} faces.")
        st.balloons()
