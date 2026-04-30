import streamlit as st
import streamlit.components.v1 as components
from rag_chat import (
    ask_question,
    summarize_paper,
    compare_papers,
    generate_related_questions
)

PAGE_ICON = """
<svg width="96" height="96" viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="96" height="96" rx="22" fill="#EAF3DE"/>
  <rect x="22" y="16" width="38" height="48" rx="8" fill="#C0DD97"/>
  <rect x="28" y="26" width="22" height="3" rx="1.5" fill="#3B6D11"/>
  <rect x="28" y="33" width="16" height="3" rx="1.5" fill="#3B6D11"/>
  <rect x="28" y="40" width="19" height="3" rx="1.5" fill="#3B6D11"/>
  <rect x="28" y="47" width="13" height="3" rx="1.5" fill="#3B6D11"/>
  <circle cx="62" cy="56" r="18" fill="#9FE1CB"/>
  <circle cx="62" cy="56" r="11" fill="#E1F5EE"/>
  <circle cx="62" cy="56" r="5" fill="#1D9E75"/>
  <rect x="73" y="66" width="14" height="5" rx="2.5" fill="#0F6E56" transform="rotate(45 73 66)"/>
</svg>
"""

st.set_page_config(
    page_title="AI Research Assistant for ArXiv Papers",
    page_icon=PAGE_ICON,
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0
if "sources_retrieved" not in st.session_state:
    st.session_state.sources_retrieved = 0

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px;">
        <svg width="60" height="60" viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="96" height="96" rx="22" fill="#EAF3DE"/>
            <rect x="22" y="16" width="38" height="48" rx="8" fill="#C0DD97"/>
            <rect x="28" y="26" width="22" height="3" rx="1.5" fill="#3B6D11"/>
            <rect x="28" y="33" width="16" height="3" rx="1.5" fill="#3B6D11"/>
            <rect x="28" y="40" width="19" height="3" rx="1.5" fill="#3B6D11"/>
            <rect x="28" y="47" width="13" height="3" rx="1.5" fill="#3B6D11"/>
            <circle cx="62" cy="56" r="18" fill="#9FE1CB"/>
            <circle cx="62" cy="56" r="11" fill="#E1F5EE"/>
            <circle cx="62" cy="56" r="5" fill="#1D9E75"/>
            <rect x="73" y="66" width="14" height="5" rx="2.5" fill="#0F6E56" transform="rotate(45 73 66)"/>
        </svg>
        <div style="font-size:15px; font-weight:800; color:#085041; margin-top:10px; line-height:1.3;">
            Powered by RAG + LLaMA 3.3 70B
        </div>
        <div style="font-size:12px; color:#475569; margin-top:6px;">
            Ask, retrieve, compare, summarize.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1.5px solid #C0DD97; margin:0 0 16px;'>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.questions_asked = 0
        st.session_state.sources_retrieved = 0
        st.rerun()

    show_context_toggle = st.toggle("Show retrieved chunks", value=True)

    st.markdown("<hr style='border:none; border-top:1.5px solid #C0DD97; margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px; font-weight:800; color:#085041;
                margin-bottom:10px; letter-spacing:0.08em;">
        📊 SESSION STATS
    </div>
    """, unsafe_allow_html=True)

    stats_col1, stats_col2 = st.columns(2)

    with stats_col1:
        st.markdown(f"""
        <div style="background:#EAF3DE; border-radius:12px; padding:10px 8px; text-align:center;">
            <div style="font-size:22px; font-weight:800; color:#085041;">{st.session_state.questions_asked}</div>
            <div style="font-size:10px; color:#3B6D11; font-weight:600;">Questions</div>
        </div>
        """, unsafe_allow_html=True)

    with stats_col2:
        st.markdown(f"""
        <div style="background:#EAF3DE; border-radius:12px; padding:10px 8px; text-align:center;">
            <div style="font-size:22px; font-weight:800; color:#085041;">{st.session_state.sources_retrieved}</div>
            <div style="font-size:10px; color:#3B6D11; font-weight:600;">Sources</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1.5px solid #C0DD97; margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px; font-weight:800; color:#085041;
                margin-bottom:12px; letter-spacing:0.08em;">
        📈 RAGAS EVALUATION
    </div>
    """, unsafe_allow_html=True)

    ragas_scores = [
        ("Faithfulness", 0.6429),
        ("Answer Relevancy", 0.7237),
        ("Context Precision", 0.6000),
        ("Context Recall", 0.5333),
    ]

    for metric, score in ragas_scores:
        bar_pct = int(score * 100)

        if score >= 0.70:
            bar_color = "#1D9E75"
            badge_bg = "#EAF3DE"
            badge_fg = "#085041"
            badge_txt = "Good"
        elif score >= 0.55:
            bar_color = "#ca8a04"
            badge_bg = "#fef9c3"
            badge_fg = "#854d0e"
            badge_txt = "Fair"
        else:
            bar_color = "#dc2626"
            badge_bg = "#fee2e2"
            badge_fg = "#991b1b"
            badge_txt = "Low"

        st.markdown(f"""
        <div style="margin-bottom:13px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                <span style="font-size:11.5px; color:#085041; font-weight:700;">{metric}</span>
                <span style="background:{badge_bg}; color:{badge_fg}; font-size:10px;
                             font-weight:700; padding:2px 7px; border-radius:999px;">
                    {badge_txt} · {score:.2f}
                </span>
            </div>
            <div style="background:#D1FAE5; border-radius:999px; height:6px; overflow:hidden;">
                <div style="width:{bar_pct}%; background:{bar_color};
                            border-radius:999px; height:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1.5px solid #C0DD97; margin:16px 0;'>", unsafe_allow_html=True)

    with st.expander("⚙️ Tech Stack"):
        tech_groups = [
            ("Frontend", ["Streamlit"]),
            ("Backend", ["Python", "FastAPI"]),
            ("Vector Database", ["ChromaDB"]),
            ("Embeddings", ["Sentence Transformers"]),
            ("LLM", ["Groq LLaMA 3.3 70B"]),
            ("Evaluation", ["RAGAS"]),
            ("Deployment", ["Docker"]),
        ]

        for group_name, items in tech_groups:
            pills_html = "".join([
                f'<span style="background:#EAF3DE; color:#085041; font-size:10.5px; font-weight:700; '
                f'padding:3px 9px; border-radius:999px; margin:2px 2px 2px 0; display:inline-block;">{item}</span>'
                for item in items
            ])

            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="font-size:10px; color:#3B6D11; font-weight:600;
                            letter-spacing:0.05em; margin-bottom:4px;">{group_name.upper()}</div>
                {pills_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; font-size:11px; color:#3B6D11;
                padding:12px 0 4px; line-height:1.8;">
        <span style="color:#085041; font-weight:700;">ArXiv AI Research Assistant</span><br>
        Built with RAG + LLM
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f4fbf7;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

section[data-testid="stSidebar"] {
    background: #eaf7ef;
    border-right: 1.5px solid #C0DD97;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

section[data-testid="stSidebar"] .stButton > button {
    background: #EAF3DE !important;
    color: #085041 !important;
    border: 1.5px solid #C0DD97 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #C0DD97 !important;
}

.paper-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 5px solid #1D9E75;
    border-radius: 18px;
    padding: 16px;
    margin: 12px 0 8px 0;
    box-shadow: 0 4px 12px rgba(29,158,117,0.07);
}

.source-rank {
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 6px;
}

.source-title {
    color: #111827;
    font-size: 15px;
    font-weight: 800;
    line-height: 1.45;
}

.context-card {
    background: #f4fbf7;
    border: 1px solid #C0DD97;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 10px;
}

.related-card {
    background: #f4fbf7;
    border: 1px solid #C0DD97;
    border-left: 4px solid #1D9E75;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 8px;
    font-weight: 600;
    color: #111827;
}

div[data-testid="stChatInput"] > div {
    border-color: #C0DD97 !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 8px rgba(29,158,117,0.08) !important;
    background: #ffffff !important;
}

div[data-testid="stChatInput"] > div:focus-within {
    border-color: #1D9E75 !important;
    box-shadow: 0 0 0 3px #EAF3DE !important;
}

div[data-testid="stChatInput"] button {
    background-color: #EAF3DE !important;
    border-radius: 10px !important;
    border: none !important;
}

div[data-testid="stChatInput"] button:hover {
    background-color: #9FE1CB !important;
}

div[data-testid="stExpander"] summary p {
    font-weight: 700 !important;
    font-size: 1rem !important;
    color: #111827 !important;
}

.stButton > button {
    border-radius: 999px !important;
    border: 1.5px solid #C0DD97 !important;
    background: #EAF3DE !important;
    color: #085041 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

.stButton > button:hover {
    background: #C0DD97 !important;
    border-color: #1D9E75 !important;
    color: #085041 !important;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

components.html("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700;800&display=swap');
  * { font-family: 'DM Sans', sans-serif; }
</style>
<div style="background:#ffffff; border-radius:24px; padding:30px;
            border: 1.5px solid #C0DD97;
            box-shadow: 0 8px 24px rgba(29,158,117,0.08);">
    <div style="display:flex; align-items:center; gap:15px;">
        <svg width="58" height="58" viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="96" height="96" rx="22" fill="#EAF3DE"/>
            <rect x="22" y="16" width="38" height="48" rx="8" fill="#C0DD97"/>
            <rect x="28" y="26" width="22" height="3" rx="1.5" fill="#3B6D11"/>
            <rect x="28" y="33" width="16" height="3" rx="1.5" fill="#3B6D11"/>
            <rect x="28" y="40" width="19" height="3" rx="1.5" fill="#3B6D11"/>
            <rect x="28" y="47" width="13" height="3" rx="1.5" fill="#3B6D11"/>
            <circle cx="62" cy="56" r="18" fill="#9FE1CB"/>
            <circle cx="62" cy="56" r="11" fill="#E1F5EE"/>
            <circle cx="62" cy="56" r="5" fill="#1D9E75"/>
            <rect x="73" y="66" width="14" height="5" rx="2.5" fill="#0F6E56" transform="rotate(45 73 66)"/>
        </svg>
        <div>
            <div style="font-size:12px; color:#1D9E75; font-weight:700; letter-spacing:0.08em;">
                Using Retrieval Augmented Generation System
            </div>
            <div style="font-size:32px; font-weight:900; color:#085041; line-height:1.2;">
                AI Research Assistant for ArXiv Papers
            </div>
        </div>
    </div>
    <div style="margin-top:14px; color:#475569; font-size:15px; line-height:1.6;">
        Ask questions, retrieve relevant AI/ML papers, generate grounded answers,
        summarize sources, compare papers, and inspect retrieved context.
    </div>
    <div style="margin-top:16px; display:flex; flex-wrap:wrap; gap:10px;">
        <span style="background:#EAF3DE; color:#0F6E56; padding:6px 13px; border-radius:999px; font-size:13px; font-weight:700; border:1px solid #C0DD97;">RAG</span>
        <span style="background:#EAF3DE; color:#0F6E56; padding:6px 13px; border-radius:999px; font-size:13px; font-weight:700; border:1px solid #C0DD97;">ChromaDB</span>
        <span style="background:#EAF3DE; color:#0F6E56; padding:6px 13px; border-radius:999px; font-size:13px; font-weight:700; border:1px solid #C0DD97;">Sentence Transformers</span>
        <span style="background:#EAF3DE; color:#0F6E56; padding:6px 13px; border-radius:999px; font-size:13px; font-weight:700; border:1px solid #C0DD97;">Groq LLM</span>
        <span style="background:#EAF3DE; color:#0F6E56; padding:6px 13px; border-radius:999px; font-size:13px; font-weight:700; border:1px solid #C0DD97;">Summarization</span>
        <span style="background:#EAF3DE; color:#0F6E56; padding:6px 13px; border-radius:999px; font-size:13px; font-weight:700; border:1px solid #C0DD97;">Paper Comparison</span>
    </div>
</div>
""", height=240)

components.html("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700;800&display=swap');
  * { font-family: 'DM Sans', sans-serif; }
</style>
<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px;
            margin-top:20px;">
    <div style="background:#ffffff; padding:20px; border-radius:16px;
                border:1.5px solid #C0DD97;
                box-shadow: 0 4px 12px rgba(29,158,117,0.06);">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="7" stroke="#3B6D11" stroke-width="2" stroke-linecap="round"/>
            <path d="M16.5 16.5 L21 21" stroke="#3B6D11" stroke-width="2.2" stroke-linecap="round"/>
            <path d="M8 11 h6 M11 8 v6" stroke="#3B6D11" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
        <div style="font-weight:800; color:#085041; margin-top:10px; font-size:15px;">Semantic Search</div>
        <div style="color:#3B6D11; margin-top:6px; font-size:13px; line-height:1.5;">Finds relevant chunks using embeddings.</div>
    </div>

    <div style="background:#ffffff; padding:20px; border-radius:16px;
                border:1.5px solid #C0DD97;
                box-shadow: 0 4px 12px rgba(29,158,117,0.06);">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
            <rect x="3" y="3" width="10" height="13" rx="2" fill="#9FE1CB"/>
            <rect x="9" y="9" width="12" height="12" rx="2" fill="#1D9E75" opacity="0.85"/>
            <path d="M12 14 l2.5 2.5 L19 12" stroke="white" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div style="font-weight:800; color:#085041; margin-top:10px; font-size:15px;">Grounded Answers</div>
        <div style="color:#3B6D11; margin-top:6px; font-size:13px; line-height:1.5;">Answers based on retrieved context.</div>
    </div>

    <div style="background:#ffffff; padding:20px; border-radius:16px;
                border:1.5px solid #C0DD97;
                box-shadow: 0 4px 12px rgba(29,158,117,0.06);">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
            <rect x="3" y="4" width="8" height="11" rx="2" fill="#C0DD97"/>
            <rect x="7" y="7" width="8" height="11" rx="2" fill="#9FE1CB"/>
            <rect x="11" y="10" width="8" height="11" rx="2" fill="#1D9E75"/>
            <path d="M13 14 h4 M13 17 h3" stroke="white" stroke-width="1.4" stroke-linecap="round"/>
        </svg>
        <div style="font-weight:800; color:#085041; margin-top:10px; font-size:15px;">Paper Tools</div>
        <div style="color:#3B6D11; margin-top:6px; font-size:13px; line-height:1.5;">Summarize and compare papers.</div>
    </div>
</div>
""", height=180)

st.markdown("### 💡 Try example questions")

example_questions = [
    "What are transformers?",
    "Compare transformers vs RNNs",
    "Explain attention mechanism in simple terms",
    "What are hallucinations in LLMs?"
]

cols = st.columns(2)

for i, q in enumerate(example_questions):
    with cols[i % 2]:
        if st.button(q, key=f"example_question_{i}", use_container_width=True):
            st.session_state.example_query = q

query = st.chat_input("Ask a research question...")

if "example_query" in st.session_state:
    query = st.session_state.example_query
    del st.session_state.example_query

USER_AVATAR = "https://api.dicebear.com/7.x/thumbs/svg?seed=user&backgroundColor=C0DD97&shapeColor=3B6D11"
ASSISTANT_AVATAR = "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=arxiv&backgroundColor=d1fae5"

for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

ICON_CONFIDENCE = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none"
  style="vertical-align:middle; margin-right:8px;">
  <rect x="3" y="14" width="3" height="7" rx="1.5" fill="#1D9E75" opacity="0.35"/>
  <rect x="8" y="10" width="3" height="11" rx="1.5" fill="#1D9E75" opacity="0.55"/>
  <rect x="13" y="6" width="3" height="15" rx="1.5" fill="#1D9E75" opacity="0.75"/>
  <rect x="18" y="2" width="3" height="19" rx="1.5" fill="#1D9E75"/>
</svg>"""

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.questions_asked += 1

    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Searching papers..."):
            answer, sources, score, context_docs = ask_question(query)

            st.session_state.sources_retrieved += len(sources)

            st.markdown(
                "<div style='font-size:1.15rem;font-weight:700;color:#111827;margin-bottom:4px;'>Answer</div>",
                unsafe_allow_html=True
            )
            st.write(answer)

            related_questions = generate_related_questions(query, answer)

            if related_questions:
                with st.expander("💡 Related Questions"):
                    for rq in related_questions:
                        st.markdown(
                            f"<div class='related-card'>{rq}</div>",
                            unsafe_allow_html=True
                        )

            st.markdown(
                f"<div style='display:flex;align-items:center;margin-top:16px;margin-bottom:4px;'>"
                f"{ICON_CONFIDENCE}"
                f"<span style='font-size:1.15rem;font-weight:700;color:#111827;'>Confidence</span></div>",
                unsafe_allow_html=True
            )

            st.progress(min(score, 1.0))
            st.write(round(score, 3))

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Confidence", round(score, 3))
            with metric_col2:
                st.metric("Sources Found", len(sources))
            with metric_col3:
                st.metric("Context Chunks", len(context_docs))

            if sources:
                with st.expander("Sources", icon=":material/article:"):
                    for i, s in enumerate(sources[:5]):
                        title = s.get("title", "Untitled Paper")
                        pdf_url = s.get("pdf_url", "")
                        relevance = s.get("relevance_score", 0)

                        if relevance >= 0.7:
                            relevance_color = "#15803d"
                            relevance_label = "High"
                        elif relevance >= 0.4:
                            relevance_color = "#ca8a04"
                            relevance_label = "Medium"
                        else:
                            relevance_color = "#64748b"
                            relevance_label = "Low"

                        st.markdown(f"""
                        <div class='paper-card'>
                            <div class='source-rank' style='color:{relevance_color};'>
                                Source {i + 1} &nbsp;•&nbsp; Relevance: {relevance} ({relevance_label})
                            </div>
                            <div class='source-title'>{title}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2, col3 = st.columns([1.2, 1.3, 5])

                        with col1:
                            if pdf_url:
                                st.link_button("Open Paper", pdf_url)

                        with col2:
                            if st.button("Summarize", key=f"summarize_{i}_{title}"):
                                with st.spinner("Summarizing..."):
                                    st.info(summarize_paper(title, pdf_url))

                    if len(sources) > 5:
                        st.caption(f"Showing top 5 of {len(sources)} retrieved sources.")

                    if len(sources) >= 2:
                        st.divider()
                        if st.button("Compare Top 2 Papers"):
                            with st.spinner("Comparing top papers..."):
                                st.success(compare_papers(sources[0], sources[1]))
            else:
                st.info("No strong relevant sources found for this question.")

            if show_context_toggle:
                with st.expander("🔎 Retrieved Context"):
                    if not context_docs:
                        st.write("No clean retrieved context available.")
                    else:
                        for i, c in enumerate(context_docs[:3]):
                            text = c.get("text", "")[:400]
                            title = c.get("title", "Untitled Paper")

                            st.markdown(f"""
                            <div class='context-card'>
                                <b>Chunk {i + 1}</b><br>
                                <b>Paper:</b> {title}<br><br>
                                {text}...
                            </div>
                            """, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("<div class='footer'>About this project: RAG-powered assistant for searching, answering, summarizing, comparing, and evaluating ArXiv AI/ML papers.</div>", unsafe_allow_html=True)