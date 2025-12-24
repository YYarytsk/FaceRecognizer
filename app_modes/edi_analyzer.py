from __future__ import annotations

import streamlit as st

from app_shared import (
    basic_837_checks,
    load_837_text,
    load_text_from_upload,
    summarize_837_with_llm,
    truncate_text,
)


def _ensure_state_defaults() -> None:
    defaults = {
        "edi_guides": [],
        "edi_processed_guides": {},
        "edi_samples": [],
        "edi_processed_samples": {},
        "edi_last_result": None,
        "edi_last_text": None,
        "edi_variant_choice": "Auto detect",
    }
    for key, value in defaults.items():
        if key in st.session_state:
            continue
        if isinstance(value, dict):
            st.session_state[key] = value.copy()
        else:
            st.session_state[key] = value


def render() -> None:
    st.header("📄 837 File Analyzer")
    st.write(
        "Upload reference guides to ground the validator, then analyze individual 837 EDI claim files for structural issues."
    )

    _ensure_state_defaults()

    with st.expander("📚 Upload 837 Implementation Guides", expanded=not st.session_state.edi_guides):
        st.write(
            "Add reference guides (PDF, DOCX, TXT, or Markdown) that explain the 837 standard. The analyzer uses the most recent uploads when forming its report."
        )
        guide_files = st.file_uploader(
            "Upload guide files",
            type=["pdf", "docx", "txt", "md"],
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
                    extracted = load_text_from_upload(uploaded)
                except Exception as err:
                    st.warning(f"Failed to read {uploaded.name}: {err}")
                    continue
                snippet = truncate_text(extracted, 12000)
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

    with st.expander("📦 Upload sample 837 files", expanded=not st.session_state.edi_samples):
        st.write(
            "Add representative claim files (EDI, TXT, DOCX, or PDF) to give the assistant concrete examples of well-formed data."
        )
        sample_files = st.file_uploader(
            "Upload sample claims",
            type=["txt", "edi", "dat", "pdf", "x12", "docx"],
            accept_multiple_files=True,
            key="edi_sample_uploader",
        )
        if sample_files:
            added_samples = 0
            for uploaded in sample_files:
                file_key = f"sample::{uploaded.name}:{getattr(uploaded, 'size', 0)}"
                if st.session_state.edi_processed_samples.get(file_key):
                    continue
                try:
                    extracted = load_837_text(uploaded)
                except Exception as err:
                    st.warning(f"Failed to read {uploaded.name}: {err}")
                    continue
                snippet = truncate_text(extracted, 8000)
                st.session_state.edi_samples.append({"name": uploaded.name, "content": snippet})
                st.session_state.edi_processed_samples[file_key] = True
                added_samples += 1
            if added_samples:
                st.success(f"Loaded {added_samples} sample file{'s' if added_samples != 1 else ''}.")
        if st.session_state.edi_samples:
            st.caption("Recent samples:")
            for sample in reversed(st.session_state.edi_samples[-5:]):
                st.write(f"• {sample['name']} — {len(sample['content'])} chars cached")
            if st.button("Clear sample library", type="secondary", key="clear_edi_samples"):
                st.session_state.edi_samples = []
                st.session_state.edi_processed_samples = {}
                st.toast("Cleared stored claim samples.")

    st.divider()
    st.subheader("Analyze an 837 File")
    st.write(
        "Upload an 837 EDI (.txt/.edi/.dat) or PDF export. The analyzer runs heuristic checks and summarizes findings with the LLM."
    )
    variant_option = st.radio(
        "Validation profile",
        options=["Auto detect", "Force 837I (Institutional)", "Force 837P (Professional)"],
        key="edi_variant_choice",
        horizontal=True,
    )
    forced_variant = None
    if variant_option == "Force 837I (Institutional)":
        forced_variant = "institutional"
    elif variant_option == "Force 837P (Professional)":
        forced_variant = "professional"
    edi_upload = st.file_uploader(
        "Upload 837 file",
        type=["txt", "edi", "dat", "pdf", "x12"],
        accept_multiple_files=False,
        key="edi_claim_uploader",
    )
    if edi_upload is not None:
        try:
            edi_text = load_837_text(edi_upload)
        except Exception as err:
            st.error(f"Unable to read uploaded file: {err}")
            edi_text = None
        if edi_text:
            st.session_state.edi_last_text = edi_text
            checks = basic_837_checks(edi_text, forced_variant=forced_variant)
            st.session_state.edi_last_result = {
                "filename": edi_upload.name,
                "issues": checks.get("issues", []),
                "transactions": checks.get("transactions", []),
                "metadata": checks.get("metadata", {}),
                "suggestions": checks.get("suggestions", []),
            }
            st.success(
                f"File '{edi_upload.name}' loaded. {len(st.session_state.edi_last_result['transactions'])} transaction set"
                f"{'s' if len(st.session_state.edi_last_result['transactions']) != 1 else ''} detected."
            )

    last_result = st.session_state.get("edi_last_result")
    last_text = st.session_state.get("edi_last_text")
    if not last_result or not last_text:
        return

    st.markdown(f"**Analyzed file:** `{last_result['filename']}`")
    metadata = last_result.get("metadata") or {}
    if metadata:
        detected_version = metadata.get("detected_version") or "n/a"
        detected_variant = metadata.get("detected_variant") or "unknown"
        active_variant = metadata.get("active_variant") or "unknown"
        forced = metadata.get("forced_variant") or "auto"
        st.caption(
            f"Detected version: {detected_version} · Detected profile: {detected_variant} · Active validation: {active_variant} ({forced})"
        )
        if metadata.get("variant_mismatch"):
            st.warning(
                "Forced profile differs from the guide version detected in the file. Double-check the selection."
            )
    if last_result["issues"]:
        st.error("Heuristic findings:")
        for item in last_result["issues"]:
            st.write(f"- {item}")
    else:
        st.success("No structural issues detected by heuristic checks.")

    suggestions = last_result.get("suggestions") or []
    if suggestions:
        st.info("Suggested fixes:")
        for tip in suggestions:
            st.write(f"- {tip}")

    if last_result["transactions"]:
        with st.expander("Transaction summary", expanded=False):
            for tx in last_result["transactions"]:
                st.write(
                    f"• Control `{tx['control'] or 'n/a'}` — ST ID `{tx['set_id'] or 'n/a'}`, {tx['segment_count']} segment(s)"
                )

    guide_notes = [item["content"] for item in st.session_state.edi_guides]
    sample_notes = [item["content"] for item in st.session_state.edi_samples]
    summary = summarize_837_with_llm(last_text, last_result["issues"], guide_notes, sample_notes)
    if summary:
        st.info(summary)
    else:
        st.warning("LLM summary not available. Review the heuristic findings above.")

    with st.expander("Show raw snippet", expanded=False):
        st.code(last_text[:4000] + ("..." if len(last_text) > 4000 else ""))
