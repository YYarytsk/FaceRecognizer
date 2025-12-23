# app.py - Streamlit Face Recognition with LLM Assistant

import os
import pickle
import random
import subprocess
import time
import wave
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pyttsx3
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from facenet_pytorch import InceptionResnetV1, MTCNN
from pypdf import PdfReader
from scipy.spatial.distance import cosine
from transformers import AutoModelForCausalLM, pipeline
from transformers.models.auto.tokenization_auto import AutoTokenizer

# -----------------------------
# Configuration and Audio Assets
# -----------------------------
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SENTRY_NAME = "ADD_NAME"
ANNOUNCE_COOLDOWN_SECONDS = 45.0
ASSETS_DIR = Path("assets")
SOUND_PRESETS = {
    "DingDong": "dingdong.wav",
    "Alarm": "alarm.wav",
    "Siren": "siren.wav",
    "Air raid siren": "air_raid.wav",
}


def _generate_chime_wave(sample_rate: int) -> np.ndarray:
    duration = 0.8
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone_a = np.sin(2 * np.pi * 784 * t)
    tone_b = np.sin(2 * np.pi * 988 * t) * np.exp(-3 * t)
    waveform = 0.6 * np.concatenate([tone_a[: int(len(tone_a) * 0.45)], tone_b[: int(len(tone_b) * 0.45)]])
    return np.int16(np.clip(waveform, -1.0, 1.0) * 32767)


def _generate_alarm_wave(sample_rate: int) -> np.ndarray:
    duration = 1.4
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    base = np.sin(2 * np.pi * 1100 * t)
    mod = 0.5 * np.sin(2 * np.pi * 6 * t)
    waveform = (base * (0.6 + mod)).clip(-1.0, 1.0)
    return np.int16(waveform * 32767)


def _generate_siren_wave(sample_rate: int) -> np.ndarray:
    duration = 1.8
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sweep = 0.6 * np.sin(2 * np.pi * 1.2 * t)
    freq = 700 + 400 * sweep
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    waveform = np.sin(phase) * np.clip(t / 0.25, 0.0, 1.0) * np.clip((duration - t) / 0.35, 0.0, 1.0)
    return np.int16(np.clip(waveform, -1.0, 1.0) * 32767)


def _generate_air_raid_wave(sample_rate: int) -> np.ndarray:
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wobble = np.sin(2 * np.pi * 0.6 * t)
    freq = 500 + 520 * wobble
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    waveform = np.sin(phase) * (0.5 + 0.5 * np.sin(2 * np.pi * 8 * t))
    return np.int16(np.clip(waveform, -1.0, 1.0) * 32767)


SOUND_GENERATORS = {
    "DingDong": _generate_chime_wave,
    "Alarm": _generate_alarm_wave,
    "Siren": _generate_siren_wave,
    "Air raid siren": _generate_air_raid_wave,
}


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=1)
def _init_voice_engine() -> pyttsx3.Engine:
    engine = pyttsx3.init()
    engine.setProperty("rate", 185)
    engine.setProperty("volume", 0.85)
    return engine


def _speak_phrase(text: str) -> None:
    phrase = text.strip()
    if not phrase:
        return
    try:
        engine = _init_voice_engine()
        engine.say(phrase)
        engine.runAndWait()
    except Exception:
        pass


def _ensure_alert_sound(label: str) -> Optional[Path]:
    filename = SOUND_PRESETS.get(label)
    generator = SOUND_GENERATORS.get(label)
    if not filename or not generator:
        return None
    try:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        target = (ASSETS_DIR / filename).resolve()
        if target.exists():
            return target
        sample_rate = 16000
        pcm = generator(sample_rate)
        with wave.open(str(target), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return target
    except Exception:
        return None


def _play_alert_sound(label: str) -> None:
    sound_path = _ensure_alert_sound(label)
    if not sound_path:
        return
    try:
        subprocess.run(["afplay", str(sound_path)], check=False)
    except Exception:
        pass


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
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix in {".txt", ".md"}:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    if suffix == ".docx":
        document = Document(uploaded_file)
        return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    raise ValueError(f"Unsupported file type: {suffix}")


def _truncate_text(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def _select_relevant_sources(question: str, sources: list, top_k: int = 2) -> list:
    if not sources:
        return []
    tokens = [tok for tok in question.lower().split() if len(tok) > 3]
    scored = []
    for entry in sources:
        content = entry["content"].lower()
        score = sum(content.count(tok) for tok in tokens)
        if score:
            scored.append((score, entry))
    if not scored:
        return sources[-top_k:]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]


def _split_segments_with_positions(edi_text: str) -> list[tuple[str, int]]:
    text = edi_text.replace("\r", "")
    segments: list[tuple[str, int]] = []
    buffer: list[str] = []
    buffer_start = 1
    line_no = 1
    for raw_line in text.split("\n"):
        cursor = 0
        while True:
            tilde_idx = raw_line.find("~", cursor)
            if tilde_idx == -1:
                chunk = raw_line[cursor:]
                if chunk:
                    if not buffer:
                        buffer_start = line_no
                    buffer.append(chunk)
                break
            chunk = raw_line[cursor:tilde_idx]
            if chunk or buffer:
                if not buffer:
                    buffer_start = line_no
                buffer.append(chunk)
            segment_text = "".join(buffer).strip()
            if segment_text:
                segments.append((segment_text, buffer_start))
            buffer = []
            cursor = tilde_idx + 1
        line_no += 1
    if buffer:
        segment_text = "".join(buffer).strip()
        if segment_text:
            segments.append((segment_text, buffer_start))
    return segments


def _basic_837_checks(edi_text: str) -> dict:
    report = {"issues": [], "transactions": []}
    normalized = edi_text.replace("\r", "")
    if not normalized.strip():
        report["issues"].append("File appears to be empty after removing whitespace.")
        return report
    segments = _split_segments_with_positions(normalized)
    if not segments:
        report["issues"].append("No EDI segments (terminated by '~') were found.")
        return report

    parsed_segments: list[dict[str, Any]] = []
    tag_map: dict[str, list[dict[str, Any]]] = {}
    for idx, (seg_text, seg_line) in enumerate(segments):
        parts = seg_text.split("*")
        tag = parts[0].strip() if parts else ""
        entry = {
            "tag": tag,
            "text": seg_text,
            "parts": parts,
            "line": seg_line,
            "index": idx,
        }
        parsed_segments.append(entry)
        if tag:
            tag_map.setdefault(tag, []).append(entry)

    required_tags = ("ISA", "GS", "ST", "BHT", "SE", "GE", "IEA")
    for tag in required_tags:
        if tag not in tag_map:
            report["issues"].append(f"Missing required segment: {tag}.")

    control_refs: dict[str, tuple[Optional[str], Optional[int]]] = {
        "ISA13": (None, None),
        "IEA02": (None, None),
        "GS06": (None, None),
        "GE02": (None, None),
    }

    if tag_map.get("ISA"):
        isa_entry = tag_map["ISA"][0]
        isa_parts = isa_entry["parts"]
        isa_line = isa_entry["line"]
        if len(isa_parts) < 17:
            report["issues"].append(
                f"Line {isa_line}: ISA segment should contain 16 data elements (17 entries with tag)."
            )
        if len(isa_parts) > 13:
            control_refs["ISA13"] = (isa_parts[13], isa_line)

    if tag_map.get("IEA"):
        iea_entry = tag_map["IEA"][0]
        iea_parts = iea_entry["parts"]
        iea_line = iea_entry["line"]
        if len(iea_parts) < 3:
            report["issues"].append(
                f"Line {iea_line}: IEA segment must contain two data elements (IEA01/IEA02)."
            )
        if len(iea_parts) > 2:
            control_refs["IEA02"] = (iea_parts[2], iea_line)

    if tag_map.get("GS"):
        gs_entry = tag_map["GS"][0]
        gs_parts = gs_entry["parts"]
        gs_line = gs_entry["line"]
        if len(gs_parts) < 9:
            report["issues"].append(
                f"Line {gs_line}: GS segment should contain 8 data elements (9 entries with tag)."
            )
        if len(gs_parts) > 6:
            control_refs["GS06"] = (gs_parts[6], gs_line)

    if tag_map.get("GE"):
        ge_entry = tag_map["GE"][0]
        ge_parts = ge_entry["parts"]
        ge_line = ge_entry["line"]
        if len(ge_parts) < 3:
            report["issues"].append(
                f"Line {ge_line}: GE segment must contain two data elements (GE01/GE02)."
            )
        if len(ge_parts) > 2:
            control_refs["GE02"] = (ge_parts[2], ge_line)

    isa_value, isa_line = control_refs["ISA13"]
    iea_value, iea_line = control_refs["IEA02"]
    if isa_value and iea_value and isa_value != iea_value:
        report["issues"].append(
            f"Lines {isa_line}/{iea_line}: ISA control number {isa_value} does not match IEA control number {iea_value}."
        )

    gs_value, gs_line = control_refs["GS06"]
    ge_value, ge_line = control_refs["GE02"]
    if gs_value and ge_value and gs_value != ge_value:
        report["issues"].append(
            f"Lines {gs_line}/{ge_line}: GS control number {gs_value} does not match GE control number {ge_value}."
        )

    current_tx: Optional[dict[str, Any]] = None
    for entry in parsed_segments:
        tag = entry["tag"]
        if tag == "ST":
            if current_tx is not None:
                st_entry = current_tx["start"]
                report["issues"].append(
                    f"Line {st_entry['line']}: ST segment missing terminating SE segment before next ST."
                )
            current_tx = {"start": entry, "segments": [entry]}
        elif tag == "SE":
            if current_tx is None:
                report["issues"].append(
                    f"Line {entry['line']}: SE segment encountered without a preceding ST segment."
                )
                continue
            current_tx["segments"].append(entry)
            st_entry = current_tx["start"]
            st_parts = st_entry["parts"]
            se_parts = entry["parts"]
            se_line = entry["line"]
            segment_count = len(current_tx["segments"])
            if len(se_parts) > 1:
                try:
                    declared_count = int(se_parts[1])
                    if declared_count != segment_count:
                        report["issues"].append(
                            f"Line {se_line}: Transaction set {st_parts[1] if len(st_parts) > 1 else ''} declares {declared_count} segments in SE01 but contains {segment_count}."
                        )
                except ValueError:
                    report["issues"].append(
                        f"Line {se_line}: SE01 should be numeric representing the segment count."
                    )
            if len(st_parts) > 2 and len(se_parts) > 2 and st_parts[2] != se_parts[2]:
                report["issues"].append(
                    f"Lines {st_entry['line']}/{se_line}: Transaction set control number mismatch (ST02 {st_parts[2]} vs SE02 {se_parts[2]})."
                )
            report["transactions"].append(
                {
                    "set_id": st_parts[1] if len(st_parts) > 1 else "",
                    "control": st_parts[2] if len(st_parts) > 2 else "",
                    "segment_count": segment_count,
                }
            )
            current_tx = None
        else:
            if current_tx is not None:
                current_tx["segments"].append(entry)

    if current_tx is not None:
        st_entry = current_tx["start"]
        report["issues"].append(
            f"Line {st_entry['line']}: ST segment missing terminating SE segment before file end."
        )

    return report


def _summarize_837_with_llm(edi_text: str, issues: list[str], guide_notes: list[str]) -> Optional[str]:
    guide_excerpt = "\n\n".join(guide_notes[-3:]) if guide_notes else ""
    truncated_edi = edi_text[:6000]
    issues_text = "\n".join(f"- {item}" for item in issues) if issues else "None identified by heuristics."
    prompt = (
        "You are validating an X12 HIPAA 837 claim file. "
        "Use the available guide excerpts when relevant. "
        "Summarize the detected problems, suggest likely resolutions, and mention missing checks explicitly.\n\n"
    )
    if guide_excerpt:
        prompt += f"Guide excerpts:\n{guide_excerpt}\n\n"
    prompt += f"Heuristic issues detected:\n{issues_text}\n\n"
    prompt += "Review the following portion of the 837 file and provide a concise validation report with actionable next steps. "
    prompt += "If information is insufficient, state what additional context is needed.\n\n"
    prompt += f"837 snippet:\n{truncated_edi}"
    try:
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=220,
            do_sample=True,
            top_p=0.85,
            temperature=0.8,
            return_full_text=False,
        )
        if not outputs:
            return None
        text = outputs[0].get("generated_text", "").strip()
        return text or None
    except Exception:
        return None


def _task_key(task: dict) -> tuple:
    return (
        task.get("category"),
        task.get("name"),
        task.get("type"),
        task.get("phrase"),
        task.get("sound"),
    )


def _enqueue_task(queue_name: str, pending_name: str, task: dict) -> bool:
    key = _task_key(task)
    pending = st.session_state[pending_name]
    if key in pending:
        return False
    st.session_state[queue_name].append(task)
    pending.add(key)
    return True


def _enqueue_announcement_task(task: dict) -> bool:
    return _enqueue_task("announcement_queue", "pending_announcements", task)


def _enqueue_joke_task(task: dict) -> bool:
    return _enqueue_task("joke_queue", "pending_jokes", task)


def _schedule_announcement(name: str, allow_generic: bool, allow_sentry: bool, sentry_sound: Optional[str]) -> bool:
    label = name.strip()
    if not label:
        return False
    readable = label.replace("_", " ")
    if label == SENTRY_NAME:
        if not allow_sentry:
            return False
        task = {
            "category": "announcement",
            "name": label,
            "type": "sentry",
            "phrase": f"Warning {readable} is coming!",
            "sound": sentry_sound or "DingDong",
        }
        return _enqueue_announcement_task(task)
    if not allow_generic:
        return False
    task = {
        "category": "announcement",
        "name": label,
        "type": "phrase",
        "phrase": readable,
    }
    return _enqueue_announcement_task(task)


def _schedule_stranger_greeting() -> bool:
    task = {
        "category": "announcement",
        "name": "Unknown",
        "type": "phrase",
        "phrase": "Hello stranger!",
    }
    return _enqueue_announcement_task(task)


def _schedule_joke(name: str, phrase: str) -> bool:
    cleaned = phrase.strip()
    if not cleaned:
        return False
    task = {
        "category": "joke",
        "name": name,
        "type": "phrase",
        "phrase": cleaned,
    }
    return _enqueue_joke_task(task)


def _pop_task_for_name(name: str):
    for queue_name, pending_attr in (("announcement_queue", "pending_announcements"), ("joke_queue", "pending_jokes")):
        queue = st.session_state[queue_name]
        for idx, task in enumerate(queue):
            if task["name"] == name:
                removed = queue.pop(idx)
                st.session_state[pending_attr].discard(_task_key(removed))
                return removed
    return None


def _execute_audio_task(task: dict) -> None:
    name = task.get("name", "")
    task_type = task.get("type")
    if task_type == "sentry":
        _speak_phrase(task.get("phrase", ""))
        _play_alert_sound(task.get("sound", "DingDong"))
        st.session_state.last_announced[name] = time.time()
        return
    if task["category"] == "announcement":
        if task_type == "phrase":
            _speak_phrase(task.get("phrase", ""))
        elif task_type == "alert":
            _play_alert_sound(task.get("sound", "DingDong"))
        st.session_state.last_announced[name] = time.time()
        return
    if task["category"] == "joke":
        if task_type == "phrase":
            _speak_phrase(task.get("phrase", ""))
        st.session_state.last_joke[name] = time.time()


def _process_audio_tasks(active_names: list[str]) -> None:
    active_unique = list(dict.fromkeys(active_names))
    active_set = set(active_unique)

    def _prune(queue_name: str, pending_attr: str) -> None:
        queue = st.session_state[queue_name]
        if not queue:
            return
        kept = []
        for task in queue:
            candidate = task["name"]
            if candidate in active_set or (candidate == "Unknown" and "Unknown" in active_set):
                kept.append(task)
            else:
                st.session_state[pending_attr].discard(_task_key(task))
        queue[:] = kept

    _prune("announcement_queue", "pending_announcements")
    _prune("joke_queue", "pending_jokes")

    if not active_unique:
        st.session_state.queue_cycle_index = 0
        return

    if st.session_state.queue_cycle_index >= len(active_unique):
        st.session_state.queue_cycle_index = 0

    for offset in range(len(active_unique)):
        idx = (st.session_state.queue_cycle_index + offset) % len(active_unique)
        name = active_unique[idx]
        task = _pop_task_for_name(name)
        if task:
            _execute_audio_task(task)
            st.session_state.queue_cycle_index = (idx + 1) % len(active_unique)
            return


def _generate_joke_for_person(name: str) -> Optional[str]:
    friendly_name = name.replace("_", " ")
    prompt = (
        f"Tell a short, friendly office-appropriate joke that includes the name {friendly_name}. "
        "Keep it under 25 words."
    )
    try:
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            return_full_text=False,
        )
    except Exception:
        return None
    if not outputs:
        return None
    text = outputs[0].get("generated_text", "").strip()
    if not text:
        return None
    return text.split("\n", 1)[0].strip().strip('"')


def _generate_stranger_joke() -> Optional[str]:
    prompt = "Tell a short, welcoming office-friendly joke about meeting a mysterious stranger. Keep it under 25 words."
    try:
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            return_full_text=False,
        )
    except Exception:
        return None
    if not outputs:
        return None
    text = outputs[0].get("generated_text", "").strip()
    if not text:
        return None
    return text.split("\n", 1)[0].strip().strip('"')


def _select_personal_phrase(name: str) -> Optional[str]:
    fact_blob = st.session_state.personal_facts.get(name, "")
    options = [line.strip() for line in fact_blob.splitlines() if line.strip()]
    if options:
        return random.choice(options)
    cached = st.session_state.generated_jokes.get(name)
    if cached:
        return cached
    generated = _generate_joke_for_person(name)
    if generated:
        st.session_state.generated_jokes[name] = generated
    return generated


def _select_stranger_joke() -> Optional[str]:
    cached = st.session_state.generated_jokes.get("Unknown")
    if cached:
        return cached
    generated = _generate_stranger_joke()
    if generated:
        st.session_state.generated_jokes["Unknown"] = generated
    return generated


def _load_837_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix in {".pdf"}:
        return _load_text_from_upload(uploaded_file)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")
    return str(raw)


# -----------------------------
# Face Database Persistence
# -----------------------------
RECOGNITION_THRESHOLD = 0.6
known_embeddings = []
known_names = []
if os.path.exists("face_db.pkl"):
    with open("face_db.pkl", "rb") as f:
        data = pickle.load(f)
        known_embeddings = data["embeddings"]
        known_names = data["names"]
        st.sidebar.success(f"Loaded {len(known_names)} known face embeddings from disk.")
else:
    st.sidebar.warning("No saved face database found. Please run 'Train Face Database' first.")

# -----------------------------
# Streamlit UI - Mode Selection
# -----------------------------
st.title("🔐 Employee Face Recognition & AI Assistant")
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

# -----------------------------
# Mode 1: Train Face Database
# -----------------------------
if mode == "Train Face Database":
    st.header("📁 Train Face Database")
    st.write("Process all images in the `employees/` folder to encode known faces.")
    st.write("Ensure each filename is the person's name and contains a single face.")
    if st.button("Train/Update Now"):
        known_embeddings = []
        known_names = []
        image_files = [f for f in os.listdir("employees") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            st.error("No images found in 'employees/' folder. Please add some face images first.")
        else:
            progress = st.progress(0)
            for idx, file in enumerate(image_files):
                img_path = os.path.join("employees", file)
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as exc:
                    st.warning(f"Could not open {file}: {exc}")
                    continue
                face_tensor = mtcnn(img)
                if face_tensor is None:
                    st.warning(f"No face detected in {file}. Skipping.")
                    continue
                embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()[0]
                embedding = embedding / np.linalg.norm(embedding)
                name = os.path.splitext(file)[0]
                known_embeddings.append(embedding)
                known_names.append(name)
                progress.progress((idx + 1) / len(image_files))
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
        defaults = {
            "last_announced": {},
            "last_joke": {},
            "enable_audio_announcements": True,
            "enable_sentry_warning": True,
            "enable_personal_fact": False,
            "enable_stranger_greeting": False,
            "personal_facts": {},
            "sentry_sound_choice": next(iter(SOUND_PRESETS)),
            "generated_jokes": {},
            "announcement_queue": [],
            "joke_queue": [],
            "pending_announcements": set(),
            "pending_jokes": set(),
            "queue_cycle_index": 0,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                if isinstance(value, dict):
                    st.session_state[key] = value.copy()
                elif isinstance(value, (list, set)):
                    st.session_state[key] = value.copy()
                else:
                    st.session_state[key] = value

        announce_col, sentry_col = st.columns(2)
        with announce_col:
            st.session_state.enable_audio_announcements = st.checkbox(
                "Name announcements",
                value=st.session_state.enable_audio_announcements,
                help="Play audio announcements for recognized individuals (except NAME toggle).",
            )
        with sentry_col:
            st.session_state.enable_sentry_warning = st.checkbox(
                "NAME warning",
                value=st.session_state.enable_sentry_warning,
                help="Play the special NAME warning and alert sound.",
            )
        extras_col, stranger_col = st.columns(2)
        with extras_col:
            st.session_state.enable_personal_fact = st.checkbox(
                "Personal facts/jokes",
                value=st.session_state.enable_personal_fact,
                help="After a recognized name, speak a configured joke or fact.",
            )
        with stranger_col:
            st.session_state.enable_stranger_greeting = st.checkbox(
                "Stranger greeting",
                value=st.session_state.enable_stranger_greeting,
                help='Say "Hello stranger!" when a face is unknown.',
            )
        sound_options = list(SOUND_PRESETS.keys())
        current_sound = st.session_state.sentry_sound_choice
        try:
            default_index = sound_options.index(current_sound)
        except ValueError:
            default_index = 0
        selected_sound = st.selectbox(
            "NAME alert sound",
            sound_options,
            index=default_index,
            help="Choose the audio clip that plays with the NAME warning.",
        )
        st.session_state.sentry_sound_choice = selected_sound
        with st.expander("🎭 Personal jokes or facts", expanded=False):
            if not known_names:
                st.caption("No recognized individuals available yet. Train the database first.")
            else:
                st.caption("Enter jokes or facts (one per line). A random line plays when the person is recognized.")
                field_map = []
                for idx, person in enumerate(known_names):
                    field_key = f"fact_input_{idx}_{person}"
                    field_map.append((person, field_key))
                    default_value = st.session_state.personal_facts.get(person, "")
                    new_value = st.text_area(
                        person,
                        value=default_value,
                        height=64,
                        key=field_key,
                    )
                    cleaned_value = new_value.strip()
                    if cleaned_value:
                        st.session_state.personal_facts[person] = cleaned_value
                        st.session_state.generated_jokes.pop(person, None)
                    else:
                        st.session_state.personal_facts.pop(person, None)
                        st.session_state.generated_jokes.pop(person, None)
                if st.button("Clear all personal facts/jokes", key="clear_personal_facts"):
                    st.session_state.personal_facts = {}
                    st.session_state.generated_jokes = {}
                    for _, field_key in field_map:
                        if field_key in st.session_state:
                            st.session_state[field_key] = ""
                    st.toast("Cleared personal facts.")

        run_camera = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])
        camera = None
        if run_camera:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                st.error("Webcam could not be opened. Check camera permissions.")
                run_camera = False
        while run_camera:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to read frame from webcam. Stopping...")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_img = Image.fromarray(frame_rgb).resize((frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2))
            boxes, probs = mtcnn.detect(small_img)
            draw = ImageDraw.Draw(frame_rgb := Image.fromarray(frame_rgb))
            current_faces: list[str] = []
            if boxes is not None:
                allow_generic_audio = st.session_state.enable_audio_announcements
                allow_sentry_audio = st.session_state.enable_sentry_warning
                allow_personal_fact = st.session_state.enable_personal_fact
                allow_stranger_greeting = st.session_state.enable_stranger_greeting
                scale_x = frame_rgb.width / small_img.width
                scale_y = frame_rgb.height / small_img.height
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                    face_img = frame_rgb.crop((x1, y1, x2, y2))
                    face_tensor = mtcnn(face_img)
                    if face_tensor is not None:
                        embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()[0]
                        embedding = embedding / np.linalg.norm(embedding)
                        distances = [cosine(embedding, emb) for emb in known_embeddings]
                        min_dist = min(distances) if distances else 1.0
                        if distances and min_dist < RECOGNITION_THRESHOLD:
                            match_index = int(np.argmin(distances))
                            name = known_names[match_index]
                        else:
                            name = "Unknown"
                    else:
                        name = "Unknown"
                    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
                    font_size = max(int((x2 - x1) * 0.22), 24)
                    font = _load_font(font_size)
                    try:
                        left, top, right, bottom = draw.textbbox((0, 0), name, font=font)
                        text_width = right - left
                        text_height = bottom - top
                    except AttributeError:
                        text_width, text_height = font.getsize(name)
                    text_x = max(int(x1), 0)
                    if text_x + text_width > frame_rgb.width:
                        text_x = max(frame_rgb.width - text_width - 4, 0)
                    text_y = max(int(y1) - text_height - 6, 0)
                    draw.text((text_x, text_y), name, fill=(0, 255, 0), font=font)
                    now_ts = time.time()
                    if name != "Unknown":
                        if name not in current_faces:
                            current_faces.append(name)
                        if now_ts - st.session_state.last_announced.get(name, 0.0) >= ANNOUNCE_COOLDOWN_SECONDS:
                            _schedule_announcement(
                                name,
                                allow_generic=allow_generic_audio,
                                allow_sentry=allow_sentry_audio,
                                sentry_sound=st.session_state.sentry_sound_choice,
                            )
                        if allow_personal_fact and now_ts - st.session_state.last_joke.get(name, 0.0) >= ANNOUNCE_COOLDOWN_SECONDS:
                            personal_phrase = _select_personal_phrase(name)
                            if personal_phrase:
                                _schedule_joke(name, personal_phrase)
                    else:
                        if "Unknown" not in current_faces:
                            current_faces.append("Unknown")
                        if allow_stranger_greeting and now_ts - st.session_state.last_announced.get("Unknown", 0.0) >= ANNOUNCE_COOLDOWN_SECONDS:
                            _schedule_stranger_greeting()
                        if allow_stranger_greeting and now_ts - st.session_state.last_joke.get("Unknown", 0.0) >= ANNOUNCE_COOLDOWN_SECONDS:
                            stranger_joke = _select_stranger_joke()
                            if stranger_joke:
                                _schedule_joke("Unknown", stranger_joke)
            _process_audio_tasks(current_faces)
            FRAME_WINDOW.image(np.array(frame_rgb))
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
        img_file = st.file_uploader("Upload an image or take a photo", type=["jpg", "jpeg", "png"])
        if img_file is not None:
            base_img = Image.open(img_file).convert("RGB")
            st.image(base_img, caption="Uploaded Image", width=400)
            boxes, _ = mtcnn.detect(base_img)
            if boxes is None:
                st.warning("No face detected in the image.")
            else:
                results = []
                box_list = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    face_img = base_img.crop((x1, y1, x2, y2))
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
                    box_list.append((x1, y1, x2, y2))
                annotated = base_img.copy()
                draw = ImageDraw.Draw(annotated)
                for (x1, y1, x2, y2), name in zip(box_list, results):
                    draw.rectangle([(x1, y1), (x2, y2)], outline="magenta", width=3)
                    font_size = max(int((x2 - x1) * 0.25), 28)
                    font = _load_font(font_size)
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
                if apply_labels:
                    updated_img = base_img.copy()
                    draw_updated = ImageDraw.Draw(updated_img)
                    for (x1, y1, x2, y2), name in zip(box_list, custom_names):
                        draw_updated.rectangle([(x1, y1), (x2, y2)], outline="magenta", width=3)
                        label = name.strip() or "Unknown"
                        font_size = max(int((x2 - x1) * 0.25), 28)
                        font = _load_font(font_size)
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

# -----------------------------
# Mode 4: 837 File Analyzer
# -----------------------------
elif mode == "837 File Analyzer":
    st.header("📄 837 File Analyzer")
    st.write("Upload reference guides to ground the validator, then analyze individual 837 EDI claim files for structural issues.")
    if "edi_guides" not in st.session_state:
        st.session_state.edi_guides = []
    if "edi_processed_guides" not in st.session_state:
        st.session_state.edi_processed_guides = {}
    if "edi_last_result" not in st.session_state:
        st.session_state.edi_last_result = None
    if "edi_last_text" not in st.session_state:
        st.session_state.edi_last_text = None
    with st.expander("📚 Upload 837 Implementation Guides", expanded=not st.session_state.edi_guides):
        st.write("Add PDF guides or reference documents that explain the 837 standard. The analyzer uses the most recent uploads when forming its report.")
        guide_files = st.file_uploader(
            "Upload guide PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="edi_guide_uploader",
        )
        if guide_files:
            added_guides = 0
            for uploaded in guide_files:
                file_key = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
                if st.session_state.edi_processed_guides.get(file_key):
                    continue
                try:
                    extracted = _load_text_from_upload(uploaded)
                except Exception as err:
                    st.warning(f"Failed to read {uploaded.name}: {err}")
                    continue
                snippet = _truncate_text(extracted, 12000)
                st.session_state.edi_guides.append({"name": uploaded.name, "content": snippet})
                st.session_state.edi_processed_guides[file_key] = True
                added_guides += 1
            if added_guides:
                st.success(f"Loaded {added_guides} guide{'s' if added_guides != 1 else ''}.")
        if st.session_state.edi_guides:
            st.caption("Recent guides:")
            for guide in reversed(st.session_state.edi_guides[-5:]):
                st.write(f"• {guide['name']} — {len(guide['content'])} chars cached")
            if st.button("Clear guide library", type="secondary", key="clear_edi_guides"):
                st.session_state.edi_guides = []
                st.session_state.edi_processed_guides = {}
                st.toast("Cleared cached guide excerpts.")
    st.divider()
    st.subheader("Analyze an 837 File")
    st.write("Upload an 837 EDI (.txt/.edi/.dat) or PDF export. The analyzer runs heuristic checks and summarizes findings with the LLM.")
    edi_upload = st.file_uploader(
        "Upload 837 file",
        type=["txt", "edi", "dat", "pdf", "x12"],
        accept_multiple_files=False,
        key="edi_claim_uploader",
    )
    if edi_upload is not None:
        try:
            edi_text = _load_837_text(edi_upload)
        except Exception as err:
            st.error(f"Unable to read uploaded file: {err}")
            edi_text = None
        if edi_text:
            st.session_state.edi_last_text = edi_text
            checks = _basic_837_checks(edi_text)
            issues = checks.get("issues", [])
            transactions = checks.get("transactions", [])
            st.session_state.edi_last_result = {
                "filename": edi_upload.name,
                "issues": issues,
                "transactions": transactions,
            }
            st.success(f"File '{edi_upload.name}' loaded. {len(transactions)} transaction set{'s' if len(transactions) != 1 else ''} detected.")
    last_result = st.session_state.get("edi_last_result")
    last_text = st.session_state.get("edi_last_text")
    if last_result and last_text:
        st.markdown(f"**Analyzed file:** `{last_result['filename']}`")
        if last_result["issues"]:
            st.error("Heuristic findings:")
            for item in last_result["issues"]:
                st.write(f"- {item}")
        else:
            st.success("No structural issues detected by heuristic checks.")
        if last_result["transactions"]:
            with st.expander("Transaction summary", expanded=False):
                for tx in last_result["transactions"]:
                    st.write(
                        f"• Control `{tx['control'] or 'n/a'}` — ST ID `{tx['set_id'] or 'n/a'}`, {tx['segment_count']} segment(s)"
                    )
        guide_notes = [item["content"] for item in st.session_state.edi_guides]
        summary = _summarize_837_with_llm(last_text, last_result["issues"], guide_notes)
        if summary:
            st.info(summary)
        else:
            st.warning("LLM summary not available. Review the heuristic findings above.")
        with st.expander("Show raw snippet", expanded=False):
            st.code(last_text[:4000] + ("..." if len(last_text) > 4000 else ""))

# -----------------------------
# Mode 5: Chat with Assistant (LLM)
# -----------------------------
else:
    st.header("💬 AI Assistant Chat")
    st.write("You are now chatting with an AI assistant running locally on this machine.")
    if "history" not in st.session_state:
        st.session_state.history = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = []
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
                except Exception as err:
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
    user_input = st.text_input("Your message:", "", key="chat_input")
    if user_input:
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
        if not history_excerpt or history_excerpt[-1][0] != "User" or history_excerpt[-1][1] != pending_user:
            messages.append({"role": "user", "content": pending_user})
        try:
            prompt = llm_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
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
            except Exception as err:
                st.error(f"Assistant failed to generate a response: {err}")
                response_text = "(Assistant encountered an error while generating a response.)"
        st.session_state.history.append(("Assistant", response_text))
        if relevant_sources:
            used_sources = ", ".join(item["name"] for item in relevant_sources)
            st.info(f"Answer grounded on: {used_sources}")
        st.session_state.pending_prompt = None
    for speaker, text in st.session_state.history:
        if speaker == "User":
            st.markdown(f"**🙍 {speaker}:** {text}")
        else:
            st.markdown(f"**🤖 {speaker}:** {text}")
