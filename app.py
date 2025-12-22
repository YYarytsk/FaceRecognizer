# app.py - Streamlit Face Recognition with LLM Assistant

import os, pickle, time
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import torch
import cv2
from docx import Document
from pypdf import PdfReader
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -----------------------------
# Configuration and Model Loading
# -----------------------------
# Initialize face detection (MTCNN) and face embedding (InceptionResnetV1) models
mtcnn = MTCNN(keep_all=False)  # detect a single face per image (we assume one face per employee image)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load or initialize LLM for chat (Mistral 7B Instruct runs well on Apple Silicon)
LLM_MODEL_NAME = "lmstudio-community/Olmo-3-32B-Think-GGUF"


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"
    return "cpu"


LLM_DEVICE = _detect_device()


@st.cache_resource(show_spinner="Loading LLM model... (first run may take a minute)")
def _build_llm_pipeline(model_name: str, device_hint: str):
    tokenizer_local = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {"trust_remote_code": True}
    if device_hint == "cuda":
        model_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
    else:
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device_hint != "cuda":
        model.to(device_hint)
    if device_hint == "cuda":
        pipe_device = 0
    elif device_hint == "mps":
        pipe_device = torch.device("mps")
    else:
        pipe_device = torch.device("cpu")
    text_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer_local, device=pipe_device)
    text_pipe.model = model
    text_pipe.tokenizer = tokenizer_local
    return text_pipe


llm_pipeline = _build_llm_pipeline(LLM_MODEL_NAME, LLM_DEVICE)
if LLM_DEVICE != "cuda":
    llm_pipeline.model.config.use_cache = False
st.sidebar.info(f"LLM running on {LLM_DEVICE.upper()}")


def _load_text_from_upload(uploaded_file) -> str:
    """Extract plain text from supported uploads (txt, pdf, docx)."""
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix in {".txt", ".md"}:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        return "\n".join(pages)
    if suffix == ".docx":
        document = Document(uploaded_file)
        return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    raise ValueError(f"Unsupported file type: {suffix}")


def _truncate_text(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def _select_relevant_sources(question: str, sources: list, top_k: int = 2) -> list:
    """Return up to top_k knowledge entries most relevant to the question."""
    if not sources:
        return []
    question_tokens = [tok for tok in question.lower().split() if len(tok) > 3]
    scored = []
    for entry in sources:
        content = entry["content"].lower()
        score = sum(content.count(tok) for tok in question_tokens)
        if score:
            scored.append((score, entry))
    if not scored:
        return sources[-top_k:]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]

# Set a cosine distance threshold for recognition
RECOGNITION_THRESHOLD = 0.6

# Data structures for known faces (embeddings and names)
known_embeddings = []
known_names = []

# Attempt to load pre-computed face encodings from disk, if available
if os.path.exists("face_db.pkl"):
    with open("face_db.pkl", "rb") as f:
        data = pickle.load(f)
        known_embeddings = data["embeddings"]
        known_names = data["names"]
        st.sidebar.success(f"Loaded {len(known_names)} known face embeddings from disk.")
else:
    st.sidebar.warning("No saved face database found. Please run 'Train Face Database' first.")

# -----------------------------
# Streamlit UI - Sidebar for mode selection
# -----------------------------
st.title("🔐 Employee Face Recognition & AI Assistant")
mode = st.sidebar.selectbox("Select Mode", ["Train Face Database", "Real-Time Recognition", "Image Recognition", "Chat with Assistant"])

# -----------------------------
# Mode 1: Train Face Database
# -----------------------------
if mode == "Train Face Database":
    st.header("📁 Train Face Database")
    st.write("This will process all images in the `employees/` folder to encode known faces.")
    st.write("**Ensure** you have added/updated employee photos in the folder, one face per image, filename as the person's name.")
    if st.button("Train/Update Now"):
        known_embeddings = []
        known_names = []
        image_files = [f for f in os.listdir("employees") if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if not image_files:
            st.error("No images found in 'employees/' folder. Please add some face images first.")
        else:
            progress = st.progress(0)
            for idx, file in enumerate(image_files):
                img_path = os.path.join("employees", file)
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception as e:
                    st.warning(f"Could not open {file}: {e}")
                    continue
                # Detect face and extract aligned crop
                face_tensor = mtcnn(img)
                if face_tensor is None:
                    st.warning(f"No face detected in {file}. Skipping.")
                    continue
                # Get embedding vector
                embedding = resnet(face_tensor.unsqueeze(0))  # (1, 512) tensor
                embedding = embedding.detach().cpu().numpy()[0]  # get numpy array from tensor
                # Normalize the embedding for consistency (optional for cosine similarity)
                embedding = embedding / np.linalg.norm(embedding)
                # Save name (from filename, without extension)
                name = os.path.splitext(file)[0]
                known_embeddings.append(embedding)
                known_names.append(name)
                # update progress
                progress.progress((idx+1)/len(image_files))
            # Save to disk
            db = {"embeddings": known_embeddings, "names": known_names}
            with open("face_db.pkl", "wb") as f:
                pickle.dump(db, f)
            st.success(f"Training completed. Processed {len(known_names)} faces.")
            st.balloons()

# -----------------------------
# Mode 2: Real-Time Recognition (Webcam)
# -----------------------------
elif mode == "Real-Time Recognition":
    st.header("📷 Real-Time Face Recognition")
    if not known_embeddings:
        st.error("No known faces available. Please run 'Train Face Database' first.")
    else:
        run_camera = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])  # placeholder for video frames
        camera = None
        if run_camera:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                st.error("Webcam could not be opened. Check camera permissions.")
                run_camera = False
        # Real-time loop
        while run_camera:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to read frame from webcam. Stopping...")
                break
            # Convert frame (BGR to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Optionally, resize to speed up (e.g., 50%)
            small_img = Image.fromarray(frame_rgb).resize((frame_rgb.shape[1]//2, frame_rgb.shape[0]//2))
            # Detect faces in the frame (using MTCNN on the smaller image for speed)
            # MTCNN on PIL Image returns cropped face tensors; to get boxes, use mtcnn.detect
            boxes, probs = mtcnn.detect(small_img)
            # If faces are found, process each
            draw = ImageDraw.Draw(frame_rgb := Image.fromarray(frame_rgb))  # prepare to draw on full-res frame
            if boxes is not None:
                # Scale boxes back to original image size if we resized
                scale_x = frame_rgb.width / small_img.width
                scale_y = frame_rgb.height / small_img.height
                for box in boxes:
                    # scale coordinates
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y
                    # Crop and embed the face from the original frame
                    face_img = frame_rgb.crop((x1, y1, x2, y2))
                    face_tensor = mtcnn(face_img)  # align and resize face
                    if face_tensor is not None:
                        face_embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()[0]
                        face_embedding = face_embedding / np.linalg.norm(face_embedding)
                        # Compare with known embeddings
                        distances = [cosine(face_embedding, emb) for emb in known_embeddings]
                        min_dist = min(distances) if distances else 1.0
                        if distances and min_dist < RECOGNITION_THRESHOLD:
                            match_index = int(np.argmin(distances))
                            name = known_names[match_index]
                        else:
                            name = "Unknown"
                    else:
                        name = "Unknown"
                    # Draw bounding box and name on the frame
                    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
                    draw.text((x1, y1 - 10), name, fill=(0, 255, 0))
            # Update image in the app
            FRAME_WINDOW.image(np.array(frame_rgb))
        # Release camera if loop is exited
        if camera:
            camera.release()
        st.write("Stopped.")

# -----------------------------
# Mode 3: Image Recognition (Upload or Snap)
# -----------------------------
elif mode == "Image Recognition":
    st.header("🖼️ Face Recognition from Image")
    if not known_embeddings:
        st.error("No known faces available. Please run 'Train Face Database' first.")
    else:
        img_file = st.file_uploader("Upload an image or take a photo", type=["jpg","jpeg","png"])
        # Alternatively, one can use st.camera_input for an instant camera capture
        # img_file = st.camera_input("Take a photo")
        if img_file is not None:
            img = Image.open(img_file).convert('RGB')
            st.image(img, caption="Uploaded Image", width=400)
            # Detect faces in the image
            boxes, _ = mtcnn.detect(img)
            if boxes is None:
                st.warning("No face detected in the image.")
            else:
                draw = ImageDraw.Draw(img)
                results = []  # to collect names
                for box in boxes:
                    x1, y1, x2, y2 = box
                    face_img = img.crop((x1, y1, x2, y2))
                    face_tensor = mtcnn(face_img)
                    name = "Unknown"
                    if face_tensor is not None:
                        emb = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()[0]
                        emb = emb / np.linalg.norm(emb)
                        distances = [cosine(emb, e) for e in known_embeddings]
                        min_dist = min(distances) if distances else 1.0
                        if distances and min_dist < RECOGNITION_THRESHOLD:
                            match_index = int(np.argmin(distances))
                            name = known_names[match_index]
                    results.append(name)
                    # Draw results
                    draw.rectangle([(x1, y1), (x2, y2)], outline="magenta", width=3)
                    draw.text((x1, y1 - 10), name, fill=(255, 0, 255))
                st.image(img, caption="Recognition Result")
                st.write("Detected faces and identities:", results)

# -----------------------------
# Mode 4: Chat with Assistant (LLM)
# -----------------------------
elif mode == "Chat with Assistant":
    st.header("💬 AI Assistant Chat")
    st.write("You are now chatting with an AI assistant running locally on this machine.")
    st.write("_Feel free to ask questions. The assistant has general knowledge and can help with common queries. (No internet access)_")
    # Initialize chat history in session state
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (speaker, text) tuples
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = []  # list of {name, content}
    if "processed_sources" not in st.session_state:
        st.session_state.processed_sources = {}
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    if st.session_state.get("clear_chat_input", False):
        st.session_state.chat_input = ""
        st.session_state.clear_chat_input = False
    with st.expander("📚 Train the assistant with your documents", expanded=not st.session_state.knowledge_base):
        st.write("Upload reference files (TXT, DOCX, PDF) to ground the assistant's answers. Recent uploads are included in responses when relevant.")
        uploaded_docs = st.file_uploader(
            "Upload files",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            key="kb_uploader",
        )
        if uploaded_docs:
            added_docs = 0
            for uploaded in uploaded_docs:
                file_key = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
                if st.session_state.processed_sources.get(file_key):
                    continue
                try:
                    extracted = _load_text_from_upload(uploaded)
                except ValueError as err:
                    st.warning(str(err))
                    continue
                except Exception as err:  # pragma: no cover - defensive
                    st.warning(f"Failed to read {uploaded.name}: {err}")
                    continue
                extracted = _truncate_text(extracted)
                if not extracted.strip():
                    st.warning(f"No text extracted from {uploaded.name}.")
                    continue
                st.session_state.knowledge_base.append({"name": uploaded.name, "content": extracted})
                st.session_state.processed_sources[file_key] = True
                added_docs += 1
            if added_docs:
                st.success(f"Added {added_docs} document{'s' if added_docs > 1 else ''} to the knowledge base.")
        if st.session_state.knowledge_base:
            st.caption("Loaded sources (most recent first):")
            for item in reversed(st.session_state.knowledge_base[-5:]):
                st.write(f"• {item['name']} — {len(item['content'])} chars")
            if st.button("Clear uploaded knowledge", type="secondary"):
                st.session_state.knowledge_base = []
                st.session_state.processed_sources = {}
                st.toast("Knowledge base cleared.")
    if st.button("Clear chat history", type="secondary"):
        st.session_state.history = []
        st.toast("Chat cleared.")
    # Chat input
    user_input = st.text_input("Your message:", "", key="chat_input")
    if user_input:
        # Add user message to history
        st.session_state.history.append(("User", user_input))
        st.session_state.pending_prompt = user_input
        st.session_state.clear_chat_input = True
        st.rerun()
    pending_user = st.session_state.get("pending_prompt")
    if pending_user:
        history_excerpt = st.session_state.history[-4:]
        messages = []
        relevant_sources = _select_relevant_sources(pending_user, st.session_state.knowledge_base)
        if relevant_sources:
            knowledge_snippets = []
            for item in relevant_sources:
                snippet = item["content"][:1200]
                knowledge_snippets.append(f"{item['name']}:\n{snippet}")
            context_blob = "\n\n".join(knowledge_snippets)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Use the provided reference notes when they are relevant. "
                        "If the notes do not contain the answer, explain that clearly before offering general guidance.\n\n"
                        f"Reference notes:\n{context_blob}"
                    ),
                }
            )
        for speaker, text in history_excerpt:
            role = "user" if speaker == "User" else "assistant"
            messages.append({"role": role, "content": text})
        # ensure the latest pending user input is last in case rerun missed it
        if not history_excerpt or history_excerpt[-1][0] != "User" or history_excerpt[-1][1] != pending_user:
            messages.append({"role": "user", "content": pending_user})
        try:
            prompt = llm_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # pragma: no cover - fallback if template unavailable
            conversation_block = "\n".join(f"{speaker}: {text}" for speaker, text in history_excerpt)
            prompt = f"{conversation_block}\nAssistant:"
        with st.spinner("Assistant is typing..."):
            generation_kwargs = {
                "max_new_tokens": 160,
                "do_sample": True,
                "top_k": 50,
                "num_return_sequences": 1,
                "return_full_text": False,
            }
            if LLM_DEVICE != "cuda":
                generation_kwargs["use_cache"] = False
            try:
                outputs = llm_pipeline(prompt, **generation_kwargs)
                if not outputs:
                    response_text = "(Assistant returned no output.)"
                else:
                    response_text = outputs[0].get("generated_text", "").strip() or "(No response generated.)"
            except Exception as err:  # pragma: no cover - surface runtime errors to UI
                st.error(f"Assistant failed to generate a response: {err}")
                response_text = "(Assistant encountered an error while generating a response.)"
        st.session_state.history.append(("Assistant", response_text))
        if relevant_sources:
            used_sources = ", ".join(item["name"] for item in relevant_sources)
            st.info(f"Answer grounded on: {used_sources}")
        st.session_state.pending_prompt = None
    # Display chat history
    for speaker, text in st.session_state.history:
        if speaker == "User":
            st.markdown(f"**🙍 {speaker}:** {text}")
        else:
            st.markdown(f"**🤖 {speaker}:** {text}")
