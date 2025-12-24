from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st
import torch
from docx import Document
from facenet_pytorch import InceptionResnetV1, MTCNN
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, pipeline
from transformers.models.auto.tokenization_auto import AutoTokenizer
from PIL import ImageFont

LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
FACE_DB_PATH = Path("face_db.pkl")
EMPLOYEES_DIR = Path("employees")
RECOGNITION_THRESHOLD = 0.6


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

# Face recognition models (shared across modules)
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval()


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def load_face_database() -> tuple[list[Any], list[str]]:
    if not FACE_DB_PATH.exists():
        return [], []
    with FACE_DB_PATH.open("rb") as file:
        data = pickle.load(file)
    return data.get("embeddings", []), data.get("names", [])


def save_face_database(embeddings: list[Any], names: list[str]) -> None:
    FACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FACE_DB_PATH.open("wb") as file:
        pickle.dump({"embeddings": embeddings, "names": names}, file)


def load_text_from_upload(uploaded_file) -> str:
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


def truncate_text(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def select_relevant_sources(question: str, sources: list[dict[str, str]], top_k: int = 2) -> list[dict[str, str]]:
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


def split_segments_with_positions(edi_text: str) -> list[tuple[str, int]]:
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


def basic_837_checks(edi_text: str, forced_variant: Optional[str] = None) -> dict[str, Any]:
    report: dict[str, Any] = {"issues": [], "transactions": []}
    normalized = edi_text.replace("\r", "")
    if not normalized.strip():
        report["issues"].append("File appears to be empty after removing whitespace.")
        return report
    segments = split_segments_with_positions(normalized)
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

    # Detect guide variant from GS08/ST03 to align heuristics with FMMIS rules.
    detected_version: Optional[str] = None
    if tag_map.get("GS"):
        gs_candidate = tag_map["GS"][0]["parts"]
        if len(gs_candidate) > 8 and gs_candidate[8].strip():
            detected_version = gs_candidate[8].strip()
    if not detected_version:
        for st_entry in tag_map.get("ST", []):
            if len(st_entry["parts"]) > 3 and st_entry["parts"][3].strip():
                detected_version = st_entry["parts"][3].strip()
                break
    normalized_forced = (
        forced_variant
        if forced_variant in {"institutional", "professional"}
        else None
    )
    detected_variant: Optional[str] = None
    if detected_version:
        upper_version = detected_version.upper()
        if "X223" in upper_version:
            detected_variant = "institutional"
        elif "X222" in upper_version:
            detected_variant = "professional"
    variant = normalized_forced or detected_variant or "unknown"
    report["metadata"] = {
        "detected_version": detected_version or "",
        "detected_variant": detected_variant or "",
        "active_variant": variant,
        "forced_variant": normalized_forced or "",
        "variant_mismatch": bool(
            normalized_forced and detected_variant and normalized_forced != detected_variant
        ),
    }

    def has_segment(
        tag: str,
        element_matches: Optional[list[tuple[int, str]]] = None,
        predicate: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> bool:
        entries = tag_map.get(tag)
        if not entries:
            return False
        if not element_matches:
            if predicate is None:
                return True
        for entry in entries:
            parts = entry["parts"]
            matched = True
            if element_matches:
                for idx, expected in element_matches:
                    if len(parts) <= idx or parts[idx].strip() != expected:
                        matched = False
                        break
            if matched and predicate is not None and not predicate(entry):
                matched = False
            if matched:
                return True
        return False

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

    common_rules = [
        {
            "tag": "NM1",
            "element_matches": [(1, "41")],
            "message": "Loop 1000A submitter NM1*41 is required by the FMMIS 837 guides.",
        },
        {
            "tag": "PER",
            "element_matches": [(1, "IC")],
            "message": "Submitter contact PER*IC is required in loop 1000A per FMMIS guidance.",
        },
        {
            "tag": "NM1",
            "element_matches": [(1, "40")],
            "message": "Loop 1000B receiver NM1*40 is required by the companion guide.",
        },
        {
            "tag": "HL",
            "element_matches": [(3, "20")],
            "message": "Loop 2000A billing provider HL with code 20 is missing.",
        },
        {
            "tag": "HL",
            "element_matches": [(3, "21")],
            "message": "Loop 2000B subscriber HL with code 21 is required by FMMIS.",
        },
        {
            "tag": "NM1",
            "element_matches": [(1, "85")],
            "message": "Loop 2010AA billing provider NM1*85 must be present.",
        },
        {
            "tag": "NM1",
            "element_matches": [(1, "PR")],
            "message": "Loop 2010BB payer NM1*PR must appear for Florida Medicaid submissions.",
        },
        {
            "tag": "CLM",
            "message": "Claim information (CLM) segment is required in loop 2300.",
        },
    ]

    for rule in common_rules:
        if not has_segment(rule["tag"], rule.get("element_matches")):
            report["issues"].append(rule["message"])

    if not has_segment(
        "HI",
        predicate=lambda entry: any(part.strip().startswith(("ABK", "ABF")) for part in entry["parts"][1:]),
    ):
        report["issues"].append(
            "Principal diagnosis (HI segment with qualifier ABK/ABF) not found; FMMIS requires a principal diagnosis."
        )

    if not has_segment("REF", [(1, "F8")]):
        report["issues"].append(
            "Claim must include REF*F8 (payer claim control) per Florida Medicaid billing instructions."
        )

    if variant == "institutional":
        institutional_rules = [
            {"tag": "CL1", "message": "Institutional claims must send CL1 in loop 2300 per the 837I guide."},
            {
                "tag": "DTP",
                "element_matches": [(1, "434")],
                "message": "DTP*434 (statement from date) missing; FMMIS 837I requires it.",
            },
            {
                "tag": "DTP",
                "element_matches": [(1, "435")],
                "message": "DTP*435 (statement through date) missing for institutional claims.",
            },
            {
                "tag": "SV2",
                "message": "Service line SV2 is expected in loop 2400 for institutional claims.",
            },
        ]
        for rule in institutional_rules:
            if not has_segment(rule["tag"], rule.get("element_matches")):
                report["issues"].append(rule["message"])
    elif variant == "professional":
        professional_rules = [
            {
                "tag": "PRV",
                "element_matches": [(1, "BI")],
                "message": "Professional claims must include PRV*BI in loop 2000A per FMMIS 837P guide.",
            },
            {
                "tag": "NM1",
                "element_matches": [(1, "82")],
                "message": "Rendering provider NM1*82 required in loop 2310B for professional claims.",
            },
            {
                "tag": "SV1",
                "message": "Service line SV1 segment is required in loop 2400 for professional claims.",
            },
            {
                "tag": "DTP",
                "element_matches": [(1, "472")],
                "message": "DTP*472 (service date) missing; FMMIS 837P requires it on each claim.",
            },
        ]
        for rule in professional_rules:
            if not has_segment(rule["tag"], rule.get("element_matches")):
                report["issues"].append(rule["message"])
    else:
        if normalized_forced is None:
            report["issues"].append(
                "Unable to determine whether file targets 837I (X223A2) or 837P (X222A1); verify GS08/ST03 version values."
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

    suggestions: list[str] = []

    def add_suggestion(text: str) -> None:
        if text and text not in suggestions:
            suggestions.append(text)

    missing_prefix = "Missing required segment: "
    for issue in report["issues"]:
        if issue.startswith(missing_prefix):
            tag_name = issue[len(missing_prefix) :].rstrip(".")
            add_suggestion(
                f"Add the {tag_name} segment as defined in the 005010 guide so the envelope structure is complete."
            )
            continue
        if "ISA segment should contain 16 data elements" in issue:
            add_suggestion(
                "Review ISA01-ISA16; each element must be present and padded to the correct width before the IEA segment."
            )
        elif "IEA segment must contain two data elements" in issue:
            add_suggestion("Populate IEA01 with the functional group count and IEA02 with the ISA control number.")
        elif "GS segment should contain 8 data elements" in issue:
            add_suggestion("Fill out GS01-GS08, ensuring GS08 carries 005010X223A2 or 005010X222A1.")
        elif "GE segment must contain two data elements" in issue:
            add_suggestion("Set GE01 to the number of transaction sets and GE02 to the GS06 control number.")
        elif "ISA control number" in issue:
            add_suggestion("Make ISA13 and IEA02 match; regenerate interchange control numbers if needed.")
        elif "GS control number" in issue:
            add_suggestion("Ensure GS06 and GE02 share the same functional group control number.")
        elif "ST segment missing terminating SE segment" in issue:
            add_suggestion("Pair each ST segment with a matching SE segment and update SE01 segment counts.")
        elif "SE segment encountered without a preceding ST" in issue:
            add_suggestion("Remove stray SE segments or restore the missing ST header before them.")
        elif "declares" in issue and "SE01" in issue:
            add_suggestion("Recalculate SE01 to equal the actual segment count from ST to SE inclusive.")
        elif "Transaction set control number mismatch" in issue:
            add_suggestion("Set ST02 and SE02 to the same transaction set control number.")
        elif "Loop 1000A submitter NM1*41" in issue:
            add_suggestion("Include loop 1000A NM1*41 with the Medicaid submitter name and identifier from the companion guide.")
        elif "Submitter contact PER*IC" in issue:
            add_suggestion("Add PER*IC with contact name, phone, and email in loop 1000A so Medicaid can reach the submitter.")
        elif "Loop 1000B receiver NM1*40" in issue:
            add_suggestion("Send NM1*40 with Florida Medicaid payer identifiers in loop 1000B.")
        elif "Loop 2000A billing provider HL" in issue:
            add_suggestion("Add HL*...*20*1*1 to start loop 2000A for the billing provider hierarchy.")
        elif "Loop 2000B subscriber HL" in issue:
            add_suggestion("Ensure loop 2000B HL includes code 21 for the subscriber level per the guide.")
        elif "Loop 2010AA billing provider NM1*85" in issue:
            add_suggestion("Populate NM1*85 with billing provider identity and matching NPI/Medicaid ID.")
        elif "Loop 2010BB payer NM1*PR" in issue:
            add_suggestion("Include NM1*PR with payer ID 'FLMCD' or the value from the payer companion guide.")
        elif "Claim information (CLM) segment" in issue:
            add_suggestion("Add CLM with claim number, monetary amounts, place of service, and frequency code in loop 2300.")
        elif "Principal diagnosis" in issue:
            add_suggestion("Insert an HI segment with qualifier ABK (principal diagnosis ICD-10) and the correct diagnosis code.")
        elif "REF*F8" in issue:
            add_suggestion("Include REF*F8 with the payer claim control number assigned by Medicaid.")
        elif "Institutional claims must send CL1" in issue:
            add_suggestion("Add CL1 with patient status and visit code in loop 2300 for institutional claims.")
        elif "DTP*434" in issue:
            add_suggestion("Populate DTP*434 with the statement-from date using qualifier D8 or RD8 as required.")
        elif "DTP*435" in issue:
            add_suggestion("Populate DTP*435 with the statement-through date covering the bill period.")
        elif "Service line SV2" in issue:
            add_suggestion("Add SV2 service line segments with revenue code, HCPCS, units, and charges for each institutional line.")
        elif "PRV*BI" in issue:
            add_suggestion("Include PRV*BI in loop 2000A with taxonomy code to identify the billing provider specialty.")
        elif "Rendering provider NM1*82" in issue:
            add_suggestion("Add NM1*82 in loop 2310B with the rendering provider's NPI and name.")
        elif "Service line SV1" in issue:
            add_suggestion("Insert SV1 segments for each professional service line with procedure code, modifiers, and charge amount.")
        elif "DTP*472" in issue:
            add_suggestion("Provide DTP*472 on each professional claim to capture the service date (format D8).")
        elif "Unable to determine whether file targets" in issue:
            add_suggestion("Set GS08/ST03 to 005010X223A2 for 837I or 005010X222A1 for 837P, or choose the matching profile in the analyzer.")

    metadata = report.get("metadata") or {}
    if metadata.get("variant_mismatch"):
        add_suggestion(
            "Either adjust the forced profile selection or align GS08/ST03 with the actual claim type so validation rules match the file."
        )

    report["suggestions"] = suggestions

    return report


def summarize_837_with_llm(
    edi_text: str,
    issues: list[str],
    guide_notes: list[str],
    sample_snippets: Optional[list[str]] = None,
) -> Optional[str]:
    guide_excerpt = "\n\n".join(guide_notes[-3:]) if guide_notes else ""
    sample_excerpt = "\n\n".join(sample_snippets[-2:]) if sample_snippets else ""
    truncated_edi = edi_text[:6000]
    issues_text = "\n".join(f"- {item}" for item in issues) if issues else "None identified by heuristics."
    prompt = (
        "You are validating an X12 HIPAA 837 claim file. "
        "Use the available guide excerpts when relevant. "
        "Summarize the detected problems, suggest likely resolutions, and mention missing checks explicitly.\n\n"
    )
    if guide_excerpt:
        prompt += f"Guide excerpts:\n{guide_excerpt}\n\n"
    if sample_excerpt:
        prompt += f"Reference claim samples:\n{sample_excerpt}\n\n"
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


def load_837_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix in {".pdf", ".docx", ".txt", ".md"}:
        return load_text_from_upload(uploaded_file)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")
    return str(raw)
