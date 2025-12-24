from __future__ import annotations

import platform
import random
import subprocess
import time
import wave
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyttsx3
import streamlit as st
from PIL import Image, ImageDraw
from scipy.spatial.distance import cosine

from app_shared import (
    RECOGNITION_THRESHOLD,
    llm_pipeline,
    load_font,
    mtcnn,
    resnet,
)

SENTRY_NAME = "Mathew_Titzman"
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
    if platform.system() == "Darwin":
        try:
            subprocess.run(["say", phrase], check=False)
            return
        except Exception:
            pass
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


def _ensure_state_defaults() -> None:
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
        "run_camera_active": False,
        "camera_resource": None,
    }
    for key, value in defaults.items():
        if key in st.session_state:
            continue
        if isinstance(value, dict):
            st.session_state[key] = value.copy()
        elif isinstance(value, list):
            st.session_state[key] = value[:]
        elif isinstance(value, set):
            st.session_state[key] = value.copy()
        else:
            st.session_state[key] = value


def render() -> None:
    st.header("📷 Real-Time Face Recognition")

    embeddings = st.session_state.get("known_embeddings", [])
    names = st.session_state.get("known_names", [])
    if not embeddings:
        st.error("No known faces available. Please run 'Train Face Database' first.")
        return

    _ensure_state_defaults()

    announce_col, sentry_col = st.columns(2)
    with announce_col:
        st.session_state.enable_audio_announcements = st.checkbox(
            "Name announcements",
            value=st.session_state.enable_audio_announcements,
            help="Play audio announcements for recognized individuals (except Mathew toggle).",
        )
    with sentry_col:
        st.session_state.enable_sentry_warning = st.checkbox(
            "Mathew warning",
            value=st.session_state.enable_sentry_warning,
            help="Play the special Mathew warning and alert sound.",
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
        "Mathew alert sound",
        sound_options,
        index=default_index,
        help="Choose the audio clip that plays with the Mathew warning.",
    )
    st.session_state.sentry_sound_choice = selected_sound

    with st.expander("🎭 Personal jokes or facts", expanded=False):
        if not names:
            st.caption("No recognized individuals available yet. Train the database first.")
        else:
            st.caption("Enter jokes or facts (one per line). A random line plays when the person is recognized.")
            field_map: list[tuple[str, str]] = []
            for idx, person in enumerate(names):
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

    st.checkbox(
        "Start Webcam",
        key="run_camera_active",
        help="Toggle the live camera feed without affecting the other controls.",
    )
    run_camera = st.session_state.run_camera_active
    frame_window = st.image([])
    camera = st.session_state.get("camera_resource")

    if run_camera and camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Webcam could not be opened. Check camera permissions.")
            st.session_state.run_camera_active = False
            run_camera = False
            camera = None
        else:
            st.session_state.camera_resource = camera
    elif not run_camera and camera is not None:
        camera.release()
        st.session_state.camera_resource = None
        camera = None

    while run_camera and camera is not None:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to read frame from webcam. Stopping...")
            st.session_state.run_camera_active = False
            break

        frame_rgb_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_img = Image.fromarray(frame_rgb_array).resize(
            (frame_rgb_array.shape[1] // 2, frame_rgb_array.shape[0] // 2)
        )
        boxes, _ = mtcnn.detect(small_img)
        frame_rgb = Image.fromarray(frame_rgb_array)
        draw = ImageDraw.Draw(frame_rgb)
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
                    distances = [cosine(embedding, emb) for emb in embeddings]
                    min_dist = min(distances) if distances else 1.0
                    if distances and min_dist < RECOGNITION_THRESHOLD:
                        match_index = int(np.argmin(distances))
                        name = names[match_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"

                draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
                font_size = max(int((x2 - x1) * 0.22), 24)
                font = load_font(font_size)
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
                    if (
                        allow_personal_fact
                        and now_ts - st.session_state.last_joke.get(name, 0.0) >= ANNOUNCE_COOLDOWN_SECONDS
                    ):
                        personal_phrase = _select_personal_phrase(name)
                        if personal_phrase:
                            _schedule_joke(name, personal_phrase)
                else:
                    if "Unknown" not in current_faces:
                        current_faces.append("Unknown")
                    if (
                        allow_stranger_greeting
                        and now_ts - st.session_state.last_announced.get("Unknown", 0.0) >= ANNOUNCE_COOLDOWN_SECONDS
                    ):
                        _schedule_stranger_greeting()
                    if (
                        allow_stranger_greeting
                        and now_ts - st.session_state.last_joke.get("Unknown", 0.0) >= ANNOUNCE_COOLDOWN_SECONDS
                    ):
                        stranger_joke = _select_stranger_joke()
                        if stranger_joke:
                            _schedule_joke("Unknown", stranger_joke)

        _process_audio_tasks(current_faces)
        frame_window.image(np.array(frame_rgb))
        time.sleep(0.03)

    if not st.session_state.run_camera_active:
        if st.session_state.get("camera_resource") is not None:
            cam_obj = st.session_state.pop("camera_resource")
            try:
                cam_obj.release()
            except Exception:
                pass
        st.info("Webcam stopped.")
    else:
        st.session_state.camera_resource = camera
