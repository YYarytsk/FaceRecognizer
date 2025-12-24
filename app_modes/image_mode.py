from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from scipy.spatial.distance import cosine

from app_shared import RECOGNITION_THRESHOLD, load_font, mtcnn, resnet


def render() -> None:
    st.header("🖼️ Face Recognition from Image")

    embeddings = st.session_state.get("known_embeddings", [])
    names = st.session_state.get("known_names", [])
    if not embeddings:
        st.error("No known faces available. Please run 'Train Face Database' first.")
        return

    img_file = st.file_uploader("Upload an image or take a photo", type=["jpg", "jpeg", "png"])
    if img_file is None:
        return

    base_img = Image.open(img_file).convert("RGB")
    st.image(base_img, caption="Uploaded Image", width=400)
    boxes, _ = mtcnn.detect(base_img)
    if boxes is None:
        st.warning("No face detected in the image.")
        return

    results: list[str] = []
    box_list = []
    for box in boxes:
        x1, y1, x2, y2 = box
        face_img = base_img.crop((x1, y1, x2, y2))
        face_tensor = mtcnn(face_img)
        name = "Unknown"
        if face_tensor is not None:
            embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            distances = [cosine(embedding, emb) for emb in embeddings]
            min_dist = min(distances) if distances else 1.0
            if distances and min_dist < RECOGNITION_THRESHOLD:
                match_index = int(np.argmin(distances))
                name = names[match_index]
        results.append(name)
        box_list.append((x1, y1, x2, y2))

    annotated = base_img.copy()
    draw = ImageDraw.Draw(annotated)
    for (x1, y1, x2, y2), name in zip(box_list, results):
        draw.rectangle([(x1, y1), (x2, y2)], outline="magenta", width=3)
        font_size = max(int((x2 - x1) * 0.25), 28)
        font = load_font(font_size)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), name, font=font)
            text_width = right - left
            text_height = bottom - top
        except AttributeError:
            text_width, text_height = font.getsize(name)
        text_x = max(int(x1), 0)
        if text_x + text_width > annotated.width:
            text_x = max(annotated.width - text_width - 4, 0)
        text_y = max(int(y1) - text_height - 8, 0)
        draw.text((text_x, text_y), name, fill=(255, 0, 255), font=font)

    st.image(annotated, caption="Recognition Result")
    st.write("Detected faces and identities:", results)

    with st.form("label_faces_form"):
        st.write("Update labels shown on the image (optional):")
        custom_names = []
        for idx, name in enumerate(results):
            custom_names.append(
                st.text_input(
                    f"Name for face #{idx + 1}",
                    value=name,
                    key=f"face_label_{idx}_{img_file.name}",
                )
            )
        apply_labels = st.form_submit_button("Apply names to image")

    if not apply_labels:
        return

    updated_img = base_img.copy()
    draw_updated = ImageDraw.Draw(updated_img)
    for (x1, y1, x2, y2), name in zip(box_list, custom_names):
        draw_updated.rectangle([(x1, y1), (x2, y2)], outline="magenta", width=3)
        label = name.strip() or "Unknown"
        font_size = max(int((x2 - x1) * 0.25), 28)
        font = load_font(font_size)
        try:
            left, top, right, bottom = draw_updated.textbbox((0, 0), label, font=font)
            text_width = right - left
            text_height = bottom - top
        except AttributeError:
            text_width, text_height = font.getsize(label)
        text_x = max(int(x1), 0)
        if text_x + text_width > updated_img.width:
            text_x = max(updated_img.width - text_width - 4, 0)
        text_y = max(int(y1) - text_height - 8, 0)
        draw_updated.text((text_x, text_y), label, fill=(255, 0, 255), font=font)

    st.image(updated_img, caption="Recognition Result (Updated Names)")
    buffer = BytesIO()
    updated_img.save(buffer, format="PNG")
    st.download_button(
        "Download annotated image",
        data=buffer.getvalue(),
        file_name="recognition_result.png",
        mime="image/png",
    )
