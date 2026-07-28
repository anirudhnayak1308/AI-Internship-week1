"""Minimal Streamlit client for the live `/ingest` + `/ask` (RAG) API.

This page is a thin client only: it calls the API and renders whatever JSON
comes back. All chunking, embedding, retrieval, grounding, and generation
happen server-side in main.py - nothing is reimplemented here.

Run:
  streamlit run streamlit_app.py
"""

import json
import os

import httpx
import streamlit as st

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "https://anirudh-ai-internship.onrender.com")
MODELS = ["gpt-4o-mini", "gpt-4o", "o3-mini"]


def call_json(
    method: str, url: str, payload: dict | None = None, timeout: float = 120.0
) -> tuple[int, dict | str]:
    try:
        if method == "POST":
            response = httpx.post(url, json=payload, timeout=timeout)
        else:
            response = httpx.get(url, timeout=timeout)

        try:
            return response.status_code, response.json()
        except json.JSONDecodeError:
            return response.status_code, response.text
    except httpx.ConnectError:
        return 0, {"error": f"Cannot reach {url}."}
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def render_ask_response(data: dict | str) -> None:
    if not isinstance(data, dict) or "error" in data:
        return

    answer = data.get("answer")
    if isinstance(answer, dict):
        if answer.get("sources_needed"):
            st.warning(
                "`sources_needed: true` - the model flagged this as a refusal / "
                "insufficient context. Treat the text below as a non-answer."
            )
        else:
            st.success("`sources_needed: false` - answered from retrieved context.")

        st.markdown("### Answer")
        st.write(answer.get("answer", ""))
        st.caption(f"confidence: {answer.get('confidence')}")

    st.markdown("### Citations (retrieved chunk IDs)")
    chunk_ids = data.get("retrieved_chunk_ids") or []
    if chunk_ids:
        for chunk_id in chunk_ids:
            st.code(chunk_id, language="text")
    else:
        st.caption("No chunks were retrieved for this question.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Model", str(data.get("model", "-")))
    metric_cols[1].metric("Tokens", str(data.get("tokens_used", "-")))
    metric_cols[2].metric("Latency", f"{data.get('latency_ms', '-')} ms")
    metric_cols[3].metric("Cost", f"${data.get('cost_usd', '-')}")


st.set_page_config(page_title="RAG Demo: /ingest + /ask", layout="centered")
st.title("RAG Demo: `/ingest` + `/ask`")
st.caption(
    "Paste a document into /ingest, then ask a question against it via /ask. "
    "The API is the source of truth - this page just calls it."
)

base_url = st.sidebar.text_input("API base URL", DEFAULT_BASE_URL).rstrip("/")
st.sidebar.caption(
    "Defaults to the API_BASE_URL env var, falling back to the live Render URL. "
    "Override here to point at a local server instead - no keys are stored in this app."
)

if st.sidebar.button("Check API health"):
    status, data = call_json("GET", f"{base_url}/health", timeout=30.0)
    st.sidebar.markdown(f"**HTTP {status}**" if status else "**Not connected**")
    st.sidebar.json(data)

st.sidebar.markdown("### Run this page locally")
st.sidebar.code(
    "cd ai-engineering-bootcamp-v2/week-1v2\n"
    "source .venv/bin/activate\n"
    "streamlit run streamlit_app.py",
    language="bash",
)

tab_ingest, tab_ask = st.tabs(["Ingest a document", "Ask a question"])

with tab_ingest:
    with st.form("ingest_form"):
        document_id = st.text_input("document_id", "doc-1")
        source = st.text_input("source (optional)", "")
        text = st.text_area("Text to ingest", height=250)
        ingest_submitted = st.form_submit_button("Ingest", type="primary")

    if ingest_submitted:
        payload = {
            "document_id": document_id,
            "text": text,
            "metadata": {"source": source} if source else {},
        }
        with st.spinner(
            "Calling /ingest... (Render free tier can take ~30-50s to wake up)"
        ):
            status, data = call_json("POST", f"{base_url}/ingest", payload)
        st.markdown(f"**HTTP {status}**" if status else "**Request failed**")
        st.json(data)

with tab_ask:
    with st.form("ask_form"):
        question = st.text_area(
            "Question", "What is Retrieval-Augmented Generation?", height=100
        )
        model = st.selectbox("Model", MODELS, index=0)
        ask_submitted = st.form_submit_button("Ask", type="primary")

    if ask_submitted:
        payload = {"question": question, "model": model}
        with st.spinner(
            "Calling /ask... (Render free tier can take ~30-50s to wake up)"
        ):
            status, data = call_json("POST", f"{base_url}/ask", payload)
        st.markdown(f"**HTTP {status}**" if status else "**Request failed**")
        render_ask_response(data)
        st.markdown("### Raw JSON")
        st.json(data)
