from __future__ import annotations

import streamlit as st

from app_shared import (
    LLM_DEVICE,
    llm_pipeline,
    load_text_from_upload,
    select_relevant_sources,
    truncate_text,
)


def _ensure_state_defaults() -> None:
    defaults = {
        "history": [],
        "knowledge_base": [],
        "processed_sources": {},
        "chat_input": "",
        "pending_prompt": None,
        "clear_chat_input": False,
    }
    for key, value in defaults.items():
        if key in st.session_state:
            continue
        if isinstance(value, dict):
            st.session_state[key] = value.copy()
        else:
            st.session_state[key] = value


def render() -> None:
    st.header("💬 AI Assistant Chat")
    st.write("You are now chatting with an AI assistant running locally on this machine.")

    _ensure_state_defaults()

    with st.expander("📚 Train the assistant with your documents", expanded=not st.session_state.knowledge_base):
        st.write(
            "Upload reference files (TXT, DOCX, PDF) to ground the assistant's answers. Recent uploads are included in responses when relevant."
        )
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
                    extracted = load_text_from_upload(uploaded)
                except ValueError as err:
                    st.warning(str(err))
                    continue
                except Exception as err:
                    st.warning(f"Failed to read {uploaded.name}: {err}")
                    continue
                extracted = truncate_text(extracted)
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

    if st.session_state.get("clear_chat_input", False):
        st.session_state.chat_input = ""
        st.session_state.clear_chat_input = False

    user_input = st.text_input("Your message:", st.session_state.chat_input, key="chat_input")
    if user_input:
        st.session_state.history.append(("User", user_input))
        st.session_state.pending_prompt = user_input
        st.session_state.clear_chat_input = True
        st.rerun()

    pending_user = st.session_state.get("pending_prompt")
    if pending_user:
        history_excerpt = st.session_state.history[-4:]
        messages = []
        relevant_sources = select_relevant_sources(pending_user, st.session_state.knowledge_base)
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
