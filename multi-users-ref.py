"""
PDF 기반 멀티유저 멀티세션 RAG 챗봇 — Supabase Auth + pgvector + 다중 LLM

실행: streamlit run 7.MultiService/code/multi-users-ref.py

Secrets (Streamlit Cloud → App → Settings → Secrets, 또는 로컬 .env):
  SUPABASE_URL
  SUPABASE_ANON_KEY

OpenAI / Anthropic / Gemini API 키는 사이드바 상단에서 입력합니다.
RAG 임베딩·검색에는 OpenAI 키가 필요합니다.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
OPENAI_CHAT_MODEL = "gpt-4o-mini"
ANTHROPIC_CHAT_MODEL = "claude-3-5-haiku-20241022"
GOOGLE_CHAT_MODEL = "gemini-1.5-flash"
BATCH_SIZE = 10

ChatProvider = str  # "openai" | "anthropic" | "google"


def _css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --chat-user: #e8f4ff;
            --chat-bot: #f6f7fb;
            --accent: #1f77b4;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        }
        h1 { color: #0f172a; letter-spacing: -0.02em; }
        .rag-title { font-size: 1.85rem; font-weight: 700; color: #0f172a; }
        .rag-sub { color: #64748b; font-size: 0.95rem; margin-top: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _chunk_text(chunk: Any) -> str:
    c = getattr(chunk, "content", chunk)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                parts.append(str(getattr(block, "text", "")))
        return "".join(parts)
    return ""


def _supabase_url_key() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = (os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    return url, key


def _get_supabase() -> Client:
    url, key = _supabase_url_key()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL 및 SUPABASE_ANON_KEY(또는 SUPABASE_SERVICE_ROLE_KEY)를 "
            "Streamlit Secrets 또는 환경 변수에 설정하세요."
        )
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(url, key)
    return st.session_state.supabase_client


def _sidebar_api_keys() -> tuple[str, str, str]:
    st.sidebar.markdown("#### API 키 (멀티유저 앱)")
    o = st.sidebar.text_input("OpenAI API Key", type="password", key="mu_openai_key")
    a = st.sidebar.text_input("Anthropic API Key", type="password", key="mu_anthropic_key")
    g = st.sidebar.text_input("Google (Gemini) API Key", type="password", key="mu_gemini_key")
    return (o or "").strip(), (a or "").strip(), (g or "").strip()


def _chat_provider_select() -> ChatProvider:
    choice = st.sidebar.selectbox(
        "채팅 모델 제공자",
        ("OpenAI", "Anthropic", "Google Gemini"),
        key="mu_chat_provider",
    )
    if choice == "Anthropic":
        return "anthropic"
    if choice == "Google Gemini":
        return "google"
    return "openai"


def _build_llm(
    provider: ChatProvider,
    *,
    openai_key: str,
    anthropic_key: str,
    gemini_key: str,
    streaming: bool,
    temperature: float = 0,
):
    if provider == "openai":
        if not openai_key:
            raise RuntimeError("OpenAI API 키를 사이드바에 입력하세요.")
        return ChatOpenAI(
            api_key=openai_key,
            model=OPENAI_CHAT_MODEL,
            temperature=temperature,
            streaming=streaming,
        )
    if provider == "anthropic":
        if not anthropic_key:
            raise RuntimeError("Anthropic API 키를 사이드바에 입력하세요.")
        return ChatAnthropic(
            api_key=anthropic_key,
            model=ANTHROPIC_CHAT_MODEL,
            temperature=temperature,
            streaming=streaming,
        )
    if not gemini_key:
        raise RuntimeError("Google API 키를 사이드바에 입력하세요.")
    return ChatGoogleGenerativeAI(
        google_api_key=gemini_key,
        model=GOOGLE_CHAT_MODEL,
        temperature=temperature,
        streaming=streaming,
    )


def _openai_embeddings(api_key: str) -> OpenAIEmbeddings:
    if not api_key:
        raise RuntimeError("PDF 임베딩·RAG 검색에는 OpenAI API 키가 필요합니다.")
    return OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM, api_key=api_key)


def _format_vector_for_rpc(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def _current_user_id(sb: Client) -> str | None:
    sess = sb.auth.get_session()
    if sess is None or sess.user is None:
        return None
    return sess.user.id


def _auth_sidebar(sb: Client) -> str | None:
    st.sidebar.markdown("### 로그인 / 회원가입")
    st.sidebar.caption("로그인 ID는 Supabase Auth 이메일 주소를 사용합니다.")
    mode = st.sidebar.radio("모드", ("로그인", "회원가입"), horizontal=True, key="mu_auth_mode")
    email = st.sidebar.text_input("로그인 ID (이메일)", key="mu_auth_email")
    pw = st.sidebar.text_input("비밀번호", type="password", key="mu_auth_pw")

    if mode == "회원가입":
        pw2 = st.sidebar.text_input("비밀번호 확인", type="password", key="mu_auth_pw2")
        if st.sidebar.button("회원가입", type="primary", use_container_width=True, key="mu_signup"):
            if not email or not pw:
                st.sidebar.error("이메일과 비밀번호를 입력하세요.")
            elif pw != pw2:
                st.sidebar.error("비밀번호가 일치하지 않습니다.")
            elif len(pw) < 6:
                st.sidebar.error("비밀번호는 6자 이상이어야 합니다.")
            else:
                try:
                    sb.auth.sign_up({"email": email.strip(), "password": pw})
                    st.sidebar.success(
                        "가입 요청이 전송되었습니다. 이메일 확인이 켜져 있으면 메일을 확인한 뒤 로그인하세요."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(f"회원가입 실패: {exc}")
    else:
        if st.sidebar.button("로그인", type="primary", use_container_width=True, key="mu_signin"):
            if not email or not pw:
                st.sidebar.error("이메일과 비밀번호를 입력하세요.")
            else:
                try:
                    sb.auth.sign_in_with_password({"email": email.strip(), "password": pw})
                    st.sidebar.success("로그인되었습니다.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(f"로그인 실패: {exc}")

    uid = _current_user_id(sb)
    if uid:
        if st.sidebar.button("로그아웃", use_container_width=True, key="mu_signout"):
            sb.auth.sign_out()
            for k in (
                "mu_session_id",
                "mu_messages",
                "mu_session_row_ready",
                "mu_session_list_version",
                "mu_vectordb_open",
                "mu_session_pick_prev",
            ):
                st.session_state.pop(k, None)
            st.rerun()
    return uid


def _init_state() -> None:
    if "mu_session_id" not in st.session_state:
        st.session_state.mu_session_id = str(uuid.uuid4())
    if "mu_messages" not in st.session_state:
        st.session_state.mu_messages = []
    if "mu_session_row_ready" not in st.session_state:
        st.session_state.mu_session_row_ready = False
    if "mu_session_list_version" not in st.session_state:
        st.session_state.mu_session_list_version = 0
    if "mu_vectordb_open" not in st.session_state:
        st.session_state.mu_vectordb_open = False
    if "mu_session_pick_prev" not in st.session_state:
        st.session_state.mu_session_pick_prev = "__init__"


def _ensure_session_row(
    client: Client, user_id: str, session_id: str, title: str = "진행 중"
) -> None:
    if st.session_state.mu_session_row_ready:
        return
    client.table("chat_sessions").upsert(
        {"id": session_id, "user_id": user_id, "title": title},
        on_conflict="id",
    ).execute()
    st.session_state.mu_session_row_ready = True


def _list_sessions(client: Client, user_id: str) -> list[dict[str, Any]]:
    res = (
        client.table("chat_sessions")
        .select("id,title,created_at,user_id")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def _load_messages_from_db(client: Client, session_id: str) -> list[dict[str, str]]:
    res = (
        client.table("chat_messages")
        .select("role,content,sort_order")
        .eq("session_id", session_id)
        .order("sort_order")
        .execute()
    )
    rows = res.data or []
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def _next_sort_order(client: Client, session_id: str) -> int:
    res = (
        client.table("chat_messages")
        .select("sort_order")
        .eq("session_id", session_id)
        .order("sort_order", desc=True)
        .limit(1)
        .execute()
    )
    if not res.data:
        return 0
    return int(res.data[0]["sort_order"]) + 1


def _persist_message_pair(
    client: Client, session_id: str, user_text: str, assistant_text: str
) -> None:
    user_id = _current_user_id(client)
    if not user_id:
        return
    _ensure_session_row(client, user_id, session_id)
    so = _next_sort_order(client, session_id)
    client.table("chat_messages").insert(
        [
            {"session_id": session_id, "role": "user", "content": user_text, "sort_order": so},
            {
                "session_id": session_id,
                "role": "assistant",
                "content": assistant_text,
                "sort_order": so + 1,
            },
        ]
    ).execute()


def _delete_session_db(client: Client, session_id: str) -> None:
    client.table("chat_sessions").delete().eq("id", session_id).execute()


def _copy_vectors_for_new_session(
    client: Client, source_session_id: str, target_session_id: str
) -> None:
    res = (
        client.table("vector_documents")
        .select("file_name,content,metadata,embedding")
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = res.data or []
    batch: list[dict[str, Any]] = []
    for row in rows:
        emb = row["embedding"]
        emb_payload: Any = emb if isinstance(emb, str) else emb
        batch.append(
            {
                "session_id": target_session_id,
                "file_name": row["file_name"],
                "content": row["content"],
                "metadata": row.get("metadata") or {},
                "embedding": emb_payload,
            }
        )
        if len(batch) >= BATCH_SIZE:
            client.table("vector_documents").insert(batch).execute()
            batch.clear()
    if batch:
        client.table("vector_documents").insert(batch).execute()


def _insert_messages_snapshot(
    client: Client, target_session_id: str, messages: list[dict[str, str]]
) -> None:
    if not messages:
        return
    ins = [
        {
            "session_id": target_session_id,
            "role": m["role"],
            "content": m["content"],
            "sort_order": i,
        }
        for i, m in enumerate(messages)
    ]
    client.table("chat_messages").insert(ins).execute()


def _ingest_pdfs(
    client: Client,
    user_id: str,
    session_id: str,
    uploaded_files: list[Any],
    openai_key: str,
) -> None:
    if not uploaded_files:
        return
    _ensure_session_row(client, user_id, session_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedder = _openai_embeddings(openai_key)
    documents: list[Document] = []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            for doc in loader.load():
                doc.metadata["file_name"] = uf.name
                documents.append(doc)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    splits = splitter.split_documents(documents)
    batch: list[dict[str, Any]] = []
    for doc in splits:
        fname = doc.metadata.get("file_name") or "unknown.pdf"
        text = doc.page_content
        if not text.strip():
            continue
        vec = embedder.embed_query(text)
        if len(vec) != EMBED_DIM:
            raise RuntimeError(f"임베딩 차원 불일치: 기대 {EMBED_DIM}, 실제 {len(vec)}")
        batch.append(
            {
                "session_id": session_id,
                "file_name": fname,
                "content": text,
                "metadata": doc.metadata,
                "embedding": _format_vector_for_rpc(vec),
            }
        )
        if len(batch) >= BATCH_SIZE:
            client.table("vector_documents").insert(batch).execute()
            batch.clear()
    if batch:
        client.table("vector_documents").insert(batch).execute()


def _retrieve_docs(
    client: Client,
    session_id: str,
    question: str,
    openai_key: str,
    k: int = 4,
) -> list[Document]:
    embedder = _openai_embeddings(openai_key)
    qvec = embedder.embed_query(question)
    if len(qvec) != EMBED_DIM:
        raise RuntimeError(f"쿼리 임베딩 차원 불일치: {len(qvec)}")
    emb_str = _format_vector_for_rpc(qvec)
    docs: list[Document] = []
    try:
        res = client.rpc(
            "match_vector_documents",
            {
                "query_embedding": emb_str,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
    except Exception:
        return docs

    rows = getattr(res, "data", None)
    if not rows:
        return docs
    for row in rows:
        meta = {
            "file_name": row.get("file_name", ""),
            "id": str(row.get("id", "")),
        }
        docs.append(Document(page_content=row.get("content", ""), metadata=meta))
    return docs


def _generate_session_title(
    user_text: str,
    assistant_text: str,
    provider: ChatProvider,
    openai_key: str,
    anthropic_key: str,
    gemini_key: str,
) -> str:
    llm = _build_llm(
        provider,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        gemini_key=gemini_key,
        streaming=False,
    )
    msg = llm.invoke(
        [
            SystemMessage(
                content=(
                    "너는 대화 제목을 짓는 도우미다. 한국어로만, "
                    "40자 이내의 짧은 제목 한 줄만 출력한다. 따옴표·부연설명 금지."
                )
            ),
            HumanMessage(
                content=f"첫 질문:\n{user_text}\n\n첫 답변:\n{assistant_text}"
            ),
        ]
    )
    title = _chunk_text(msg).strip()
    title = re.sub(r'^["\'「」]|[""\'」]$', "", title).strip()
    return title[:80] or "새 세션"


def _chat_history_lc() -> list[HumanMessage | AIMessage]:
    hist: list[HumanMessage | AIMessage] = []
    for m in st.session_state.mu_messages[:-1]:
        if m["role"] == "user":
            hist.append(HumanMessage(content=m["content"]))
        else:
            hist.append(AIMessage(content=m["content"]))
    return hist


def _system_block() -> str:
    return (
        "너는 업로드된 문서를 우선 참고하여 정확하게 답하는 한국어 어시스턴트다. "
        "문서에 없으면 모른다고 말하지만, 범위를 벗어난 추측은 하지 않는다. "
        "답변 마지막에는 반드시 다음 형식으로 후속 질문 3개를 붙인다.\n\n"
        "### 추가로 해볼 만한 질문\n"
        "1. …\n"
        "2. …\n"
        "3. …"
    )


def _run_rag_stream(
    question: str,
    context_docs: list[Document],
    provider: ChatProvider,
    openai_key: str,
    anthropic_key: str,
    gemini_key: str,
) -> str:
    context = "\n\n---\n\n".join(d.page_content for d in context_docs)
    llm = _build_llm(
        provider,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        gemini_key=gemini_key,
        streaming=True,
    )
    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=_system_block()),
        *_chat_history_lc(),
        HumanMessage(
            content=f"참고 문서:\n{context}\n\n사용자 질문:\n{question}"
        ),
    ]
    out: list[str] = []
    for chunk in llm.stream(messages):
        piece = _chunk_text(chunk)
        if piece:
            out.append(piece)
    return "".join(out)


def _stream_llm(
    question: str,
    context_docs: list[Document],
    provider: ChatProvider,
    openai_key: str,
    anthropic_key: str,
    gemini_key: str,
):
    context = "\n\n---\n\n".join(d.page_content for d in context_docs)
    llm = _build_llm(
        provider,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        gemini_key=gemini_key,
        streaming=True,
    )
    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=_system_block()),
        *_chat_history_lc(),
        HumanMessage(
            content=f"참고 문서:\n{context}\n\n사용자 질문:\n{question}"
        ),
    ]
    return llm.stream(messages)


def _vectordb_file_names(client: Client, session_id: str) -> list[str]:
    res = (
        client.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    rows = res.data or []
    names = sorted({r["file_name"] for r in rows if r.get("file_name")})
    return names


def _apply_loaded_session(client: Client, session_id: str) -> None:
    st.session_state.mu_session_id = session_id
    st.session_state.mu_messages = _load_messages_from_db(client, session_id)
    st.session_state.mu_session_row_ready = True


def _session_sidebar(
    sb: Client,
    user_id: str,
    openai_key: str,
    anthropic_key: str,
    gemini_key: str,
    chat_provider: ChatProvider,
) -> None:
    sessions = _list_sessions(sb, user_id)
    id_to_label = {
        s["id"]: f"{s.get('title') or '(무제)'} — {str(s['id'])[:8]}…" for s in sessions
    }
    options = [s["id"] for s in sessions]
    select_options = [""] + options

    def _fmt_sid(sid: str) -> str:
        if not sid:
            return "— 세션을 선택하세요 —"
        return id_to_label.get(sid, sid)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 세션 선택 (변경 시 자동 로드)")
    if options:
        picked = st.selectbox(
            "저장된 세션",
            select_options,
            index=0,
            format_func=_fmt_sid,
            key="mu_session_select_box",
        )
        if picked and picked != st.session_state.mu_session_pick_prev:
            st.session_state.mu_session_pick_prev = picked
            _apply_loaded_session(sb, picked)
    else:
        st.sidebar.caption("저장된 세션이 없습니다. 대화 후 세션저장을 눌러 보세요.")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("세션로드", use_container_width=True, key="mu_btn_load"):
            sel = st.session_state.get("mu_session_select_box")
            if sel:
                st.session_state.mu_session_pick_prev = sel
                _apply_loaded_session(sb, sel)
                st.success("세션을 불러왔습니다.")
                st.rerun()
    with c2:
        if st.button("세션저장", use_container_width=True, key="mu_btn_save_snap"):
            msgs = st.session_state.mu_messages
            if len(msgs) < 2:
                st.warning("대화가 2턴 이상일 때 저장할 수 있습니다.")
            else:
                u0 = next((m for m in msgs if m["role"] == "user"), None)
                a0 = next((m for m in msgs if m["role"] == "assistant"), None)
                if not u0 or not a0:
                    st.warning("첫 사용자·어시스턴트 메시지가 필요합니다.")
                else:
                    try:
                        title = _generate_session_title(
                            u0["content"],
                            a0["content"],
                            chat_provider,
                            openai_key,
                            anthropic_key,
                            gemini_key,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"제목 생성에 실패해 기본 제목을 사용합니다: {exc}")
                        title = "새 세션"
                    new_id = str(uuid.uuid4())
                    sb.table("chat_sessions").insert(
                        {"id": new_id, "user_id": user_id, "title": title}
                    ).execute()
                    _insert_messages_snapshot(sb, new_id, st.session_state.mu_messages)
                    _copy_vectors_for_new_session(
                        sb, st.session_state.mu_session_id, new_id
                    )
                    st.session_state.mu_session_list_version += 1
                    st.success(f"새 세션으로 저장했습니다: {title}")

    c3, c4 = st.sidebar.columns(2)
    with c3:
        if st.button("세션삭제", use_container_width=True, key="mu_btn_del"):
            sel = st.session_state.get("mu_session_select_box")
            if sel:
                _delete_session_db(sb, sel)
                st.session_state.mu_session_list_version += 1
                if st.session_state.mu_session_id == sel:
                    st.session_state.mu_session_id = str(uuid.uuid4())
                    st.session_state.mu_messages = []
                    st.session_state.mu_session_row_ready = False
                st.session_state.mu_session_pick_prev = "__init__"
                st.success("삭제했습니다.")
                st.rerun()
    with c4:
        if st.button("화면초기화", use_container_width=True, key="mu_btn_reset"):
            st.session_state.mu_session_id = str(uuid.uuid4())
            st.session_state.mu_messages = []
            st.session_state.mu_session_row_ready = False
            st.session_state.mu_session_pick_prev = "__init__"
            st.rerun()

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader(
        "PDF 업로드",
        type=["pdf"],
        accept_multiple_files=True,
        key="mu_pdf_uploader",
    )
    if uploaded and st.sidebar.button("파일 처리 · 벡터 저장", key="mu_btn_ingest"):
        if not openai_key:
            st.sidebar.error("벡터 저장에는 OpenAI API 키가 필요합니다.")
        else:
            with st.spinner("PDF 분할 및 임베딩 저장 중…"):
                _ingest_pdfs(sb, user_id, st.session_state.mu_session_id, list(uploaded), openai_key)
            st.sidebar.success("Supabase 벡터 저장이 완료되었습니다.")

    st.sidebar.markdown("---")
    if st.sidebar.button("vectordb", use_container_width=True, key="mu_btn_vec"):
        st.session_state.mu_vectordb_open = not st.session_state.mu_vectordb_open


def main() -> None:
    _css()
    _init_state()

    try:
        sb = _get_supabase()
    except Exception as exc:
        st.error(str(exc))
        return

    openai_key, anthropic_key, gemini_key = _sidebar_api_keys()
    chat_provider = _chat_provider_select()

    st.sidebar.image(
        "https://streamlit.io/images/brand/streamlit-mark-color.svg",
        width=48,
    )
    st.sidebar.markdown("### 멀티유저 · 멀티세션 · 저장 / 벡터검색")

    user_id = _auth_sidebar(sb)
    if not user_id:
        st.markdown(
            '<p class="rag-title">PDF 기반 멀티유저 멀티세션 RAG 챗봇</p>',
            unsafe_allow_html=True,
        )
        st.info("사이드바에서 로그인하거나 회원가입한 뒤 이용할 수 있습니다.")
        return

    _session_sidebar(sb, user_id, openai_key, anthropic_key, gemini_key, chat_provider)

    st.markdown(
        '<p class="rag-title">PDF 기반 멀티유저 멀티세션 RAG 챗봇</p>',
        unsafe_allow_html=True,
    )
    sub = f"{chat_provider.upper()} · OpenAI 임베딩 · Supabase pgvector"
    st.markdown(f'<p class="rag-sub">{sub}</p>', unsafe_allow_html=True)

    if st.session_state.mu_vectordb_open:
        names = _vectordb_file_names(sb, st.session_state.mu_session_id)
        with st.expander("현재 세션 벡터 DB에 저장된 파일명", expanded=True):
            if names:
                for n in names:
                    st.write(f"- {n}")
            else:
                st.caption("저장된 벡터가 없습니다.")

    for m in st.session_state.mu_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("문서에 대해 질문하세요"):
        st.session_state.mu_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        docs: list[Document] = []
        if openai_key:
            try:
                docs = _retrieve_docs(sb, st.session_state.mu_session_id, prompt, openai_key)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"문서 검색 중 오류: {exc}")
        else:
            st.warning("RAG 검색을 하려면 OpenAI API 키를 사이드바에 입력하세요.")

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response: list[str] = []
            if docs:
                try:
                    stream = _stream_llm(
                        prompt,
                        docs,
                        chat_provider,
                        openai_key,
                        anthropic_key,
                        gemini_key,
                    )
                    for chunk in stream:
                        piece = _chunk_text(chunk)
                        if piece:
                            full_response.append(piece)
                            placeholder.markdown("".join(full_response) + "▌")
                    placeholder.markdown("".join(full_response))
                    answer = "".join(full_response)
                except Exception:
                    answer = _run_rag_stream(
                        prompt,
                        docs,
                        chat_provider,
                        openai_key,
                        anthropic_key,
                        gemini_key,
                    )
                    placeholder.markdown(answer)
            else:
                try:
                    llm = _build_llm(
                        chat_provider,
                        openai_key=openai_key,
                        anthropic_key=anthropic_key,
                        gemini_key=gemini_key,
                        streaming=True,
                    )
                    messages_no_ctx: list[SystemMessage | HumanMessage | AIMessage] = [
                        SystemMessage(content=_system_block()),
                        *_chat_history_lc(),
                        HumanMessage(
                            content=(
                                "참고 문서가 없거나 검색 결과가 비어 있습니다. "
                                "PDF를 업로드·처리한 뒤 다시 질문하세요.\n\n"
                                f"질문: {prompt}"
                            )
                        ),
                    ]
                    for chunk in llm.stream(messages_no_ctx):
                        piece = _chunk_text(chunk)
                        if piece:
                            full_response.append(piece)
                            placeholder.markdown("".join(full_response) + "▌")
                    placeholder.markdown("".join(full_response))
                    answer = "".join(full_response)
                except Exception as exc:  # noqa: BLE001
                    answer = f"답변 생성에 실패했습니다: {exc}"
                    placeholder.markdown(answer)

        st.session_state.mu_messages.append({"role": "assistant", "content": answer})
        _ensure_session_row(sb, user_id, st.session_state.mu_session_id)
        _persist_message_pair(sb, st.session_state.mu_session_id, prompt, answer)


if __name__ == "__main__":
    main()
