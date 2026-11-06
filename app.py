import datetime
import html
import textwrap
from typing import Any, Dict, List, Optional
import streamlit as st
import storage
import analytics
import diagnostics
import experience
import knowledge
import knowledge_store
from agent import AuraAgent
from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_STEPS
from memory import load_memory, memory_as_text
from tools import TOOL_REGISTRY

# Import presentation components
from ui import (
    STYLING_CSS,
    render_sidebar,
    render_top_bar,
    render_hero_and_cards,
    render_compact_chat_header,
    render_execution_timeline_html,
    render_chat_messages,
    render_right_insight_panel,
    render_tool_shortcuts,
    handle_prompt_submission,
    render_memory_page,
    render_knowledge_page,
    render_analytics_page,
    render_health_page,
    render_settings_page,
)

# Page configuration
st.set_page_config(
    page_title="AURA — Autonomous AI Workspace",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize Streamlit session state, storage database, and active conversation."""
    storage.initialize_database()

    if "agent" not in st.session_state:
        st.session_state.agent = AuraAgent()

    if "active_conversation_id" not in st.session_state:
        conversations = storage.list_conversations()
        if conversations:
            st.session_state.active_conversation_id = conversations[0]["id"]
        else:
            st.session_state.active_conversation_id = None

    # Messages source of truth directly from storage database (or empty for draft)
    if st.session_state.active_conversation_id:
        st.session_state.messages = storage.get_messages(st.session_state.active_conversation_id)
    else:
        st.session_state.messages = []

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "page_mode" not in st.session_state:
        st.session_state.page_mode = "chat"


def main():
    st.markdown(STYLING_CSS, unsafe_allow_html=True)
    init_session_state()

    # Sidebar Navigation Only
    render_sidebar()

    current_mode = st.session_state.get("page_mode", "chat")

    if current_mode == "analytics":
        render_analytics_page()
        return
    elif current_mode == "memory":
        render_memory_page()
        return
    elif current_mode == "knowledge":
        render_knowledge_page()
        return
    elif current_mode == "health":
        render_health_page()
        return
    elif current_mode == "settings":
        render_settings_page()
        return

    # Default View: AI Chat Application Workspace (Three-column layout matching reference)
    active_prompt = None
    if st.session_state.pending_prompt:
        active_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    uploaded_file = None

    col_main, col_right = st.columns([0.70, 0.30], gap="medium")

    with col_main:
        # Empty Chat Hero vs Active Chat Workspace
        if not st.session_state.messages:
            render_compact_chat_header()
            render_hero_and_cards()
        else:
            render_chat_messages()

        # Tool shortcuts above composer
        render_tool_shortcuts()

        # Document and image attachment expander inline near input
        with st.expander("Attach document or image (DOCX, PDF, TXT, CSV, Images)", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload document or image",
                type=["docx", "pdf", "txt", "md", "csv", "png", "jpg", "jpeg", "webp", "gif"],
                key="chat_image_attachment_uploader",
                help="Supported formats: DOCX, PDF, TXT, MD, CSV, PNG, JPG, WEBP, GIF (Max 10 MB)"
            )
            if uploaded_file is not None:
                from pathlib import Path
                from ui.icons import icon_docx, icon_file, icon_paperclip
                f_name = uploaded_file.name
                f_bytes = uploaded_file.getvalue()
                f_ext = Path(f_name).suffix.lower()
                size_kb = len(f_bytes) / 1024.0
                if f_ext == ".docx":
                    icon_svg = icon_docx(16, "#38BDF8")
                    type_label = "DOCX Document"
                elif f_ext in (".pdf", ".txt", ".md", ".csv"):
                    icon_svg = icon_file(16, "#8B9CFF")
                    type_label = f"{f_ext.replace('.', '').upper()} Document"
                else:
                    icon_svg = icon_paperclip(16, "#34D399")
                    type_label = "Image"

                st.markdown(
                    f'<div style="display:inline-flex; align-items:center; gap:8px; background:var(--bg-surface-elevated); border:1px solid var(--border-subtle); border-radius:6px; padding:6px 10px; margin-top:6px; font-size:12px; color:var(--text-primary);">'
                    f'{icon_svg} <b>{html.escape(f_name)}</b> <span style="color:var(--text-muted);">({type_label}, {size_kb:.1f} KB ready to attach)</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with col_right:
        render_right_insight_panel()

    img_bytes = None
    img_name = None
    doc_bytes = None
    doc_name = None
    if uploaded_file is not None:
        from pathlib import Path
        f_ext = Path(uploaded_file.name).suffix.lower()
        if f_ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            img_bytes = uploaded_file.getvalue()
            img_name = uploaded_file.name
        else:
            doc_bytes = uploaded_file.getvalue()
            doc_name = uploaded_file.name

    user_input = st.chat_input("Ask AURA anything...")

    if active_prompt:
        handle_prompt_submission(
            active_prompt,
            image_bytes=img_bytes,
            image_filename=img_name,
            attachment_bytes=doc_bytes,
            attachment_filename=doc_name
        )
        st.rerun()
    elif user_input:
        handle_prompt_submission(
            user_input,
            image_bytes=img_bytes,
            image_filename=img_name,
            attachment_bytes=doc_bytes,
            attachment_filename=doc_name
        )
        st.rerun()




if __name__ == "__main__":
    main()
