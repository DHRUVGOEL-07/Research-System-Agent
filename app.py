import streamlit as st
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# ── Page config (must be first) ─────────────────────────────────
st.set_page_config(
    page_title="Research System Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load env ────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0;
    }
    .main-header p {
        color: #a0aec0;
        margin: 0.5rem 0 0;
        font-size: 1rem;
    }
    .agent-step {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        border-left: 4px solid;
    }
    .step-memory    { background:#E1F5EE; border-color:#0F6E56; color:#085041; }
    .step-orchestrator { background:#E1F5EE; border-color:#0F6E56; color:#085041; }
    .step-search    { background:#E6F1FB; border-color:#185FA5; color:#0C447C; }
    .step-retrieval { background:#E6F1FB; border-color:#185FA5; color:#0C447C; }
    .step-synthesis { background:#FAECE7; border-color:#993C1D; color:#712B13; }
    .step-critic    { background:#FAEEDA; border-color:#854F0B; color:#633806; }
    .step-report    { background:#E1F5EE; border-color:#0F6E56; color:#085041; }
    .step-warn      { background:#FFF8E1; border-color:#F9A825; color:#7d5e00; }
    .report-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        line-height: 1.7;
    }
    .memory-card {
        background: #f0f4ff;
        border: 1px solid #c7d2fe;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .stat-label  { font-size: 0.8rem; color: #718096; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Import orchestrator (lazy to avoid slow startup) ────────────
@st.cache_resource
def load_agent():
    from orchestrator import build_graph, AgentState
    app = build_graph()
    return app, AgentState

# ── Memory helpers ───────────────────────────────────────────────
def load_memory_sessions():
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return []

def clear_memory():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Research Agent")
    st.markdown("---")

    # API key status
    st.markdown("### ⚙️ API Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"{'🟢' if GROQ_API_KEY else '🔴'} Groq")
        st.markdown(f"{'🟢' if TAVILY_API_KEY else '🔴'} Tavily")
    with col2:
        st.markdown(f"{'🟢' if SEMANTIC_SCHOLAR_API_KEY else '🔴'} Semantic")
        st.markdown("🟢 ArXiv")

    st.markdown("---")

    # Settings
    st.markdown("### 🎛️ Settings")
    max_papers = st.slider("Max papers per source", 3, 10, 5)
    year_from = st.slider("Papers from year", 2018, 2024, 2022)
    max_iterations = st.slider("Max critic iterations", 1, 5, 2)

    st.markdown("---")

    # Past sessions
    st.markdown("### 🧠 Memory")
    sessions = load_memory_sessions()
    if sessions:
        st.markdown(f"**{len(sessions)} past sessions**")
        for s in reversed(sessions[-5:]):
            st.markdown(f"""
            <div class="memory-card">
                <b>{s['query'][:45]}...</b><br>
                <small>{s['timestamp'][:16]}</small>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🗑️ Clear Memory", key="clear_mem_btn"):
            clear_memory()
            st.success("Memory cleared!")
    else:
        st.info("No past sessions yet")

    st.markdown("---")
    st.markdown("### 📁 Project")
    st.markdown("""
    - `orchestrator.py` — main pipeline
    - `tools/` — search tools
    - `research_db/` — vector DB
    - `memory.json` — session memory
    """)

# ── Main Header ──────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Research System Agent</h1>
    <p>Autonomous multi-agent research assistant powered by LangGraph + Groq</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ────────────────────────────────────────────────────
sessions = load_memory_sessions()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">{len(sessions)}</div>
        <div class="stat-label">Total Queries</div>
    </div>""", unsafe_allow_html=True)
with col2:
    try:
        import chromadb
        client = chromadb.PersistentClient(
            path=os.path.join(BASE_DIR, "research_db"))
        col = client.get_or_create_collection("research_papers")
        chunks = col.count()
    except:
        chunks = 0
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">{chunks}</div>
        <div class="stat-label">Papers in DB</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">3</div>
        <div class="stat-label">Search Sources</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">7</div>
        <div class="stat-label">Agent Nodes</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Query Input ──────────────────────────────────────────────────
st.markdown("### 🔍 Enter Research Query")
query = st.text_input(
    label="query",
    placeholder="e.g. latest deep learning methods for cancer detection 2024",
    label_visibility="collapsed"
)

# Quick query suggestions
st.markdown("**Quick suggestions:**")
suggestions = [
    "deep learning for medical imaging",
    "transformer models in NLP 2024",
    "reinforcement learning robotics",
    "large language model fine-tuning"
]
cols = st.columns(4)
for i, suggestion in enumerate(suggestions):
    with cols[i]:
        if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
            query = suggestion

st.markdown("<br>", unsafe_allow_html=True)

run_btn = st.button("🚀 Run Research Agent", 
                     type="primary",
                     use_container_width=True,
                     disabled=not query.strip())

# ── Agent Execution ──────────────────────────────────────────────
if run_btn and query.strip():

    # Layout: progress left, report right
    left, right = st.columns([1, 1.6])

    with left:
        st.markdown("### 🤖 Agent Progress")
        progress_bar = st.progress(0)
        status_container = st.container()

    with right:
        st.markdown("### 📄 Research Report")
        report_placeholder = st.empty()
        report_placeholder.info("⏳ Waiting for agent to complete...")

    steps_html = []

    def add_step(label, msg, style="step-search"):
        steps_html.append(
            f'<div class="agent-step {style}"><b>{label}</b> {msg}</div>'
        )
        with status_container:
            st.markdown("".join(steps_html), unsafe_allow_html=True)

    # load agent
    with st.spinner("Loading agent..."):
        try:
            app, AgentState = load_agent()
        except Exception as e:
            st.error(f"Failed to load agent: {e}")
            st.stop()

    # build initial state
    initial_state = {
        "query": query.strip(),
        "task_plan": "",
        "memory_context": "",
        "search_results": [],
        "rag_context": [],
        "synthesis": "",
        "gaps": [],
        "iteration": 0,
        "final_report": ""
    }

    add_step("🧠", "Checking memory for past sessions...", "step-memory")
    progress_bar.progress(10)

    try:
        # stream through graph nodes
        for event in app.stream(initial_state):
            node_name = list(event.keys())[0]
            node_state = event[node_name]

            if node_name == "memory":
                ctx = node_state.get("memory_context", "")
                if "No prior" in ctx:
                    add_step("🧠 Memory:", "No past sessions found", "step-memory")
                else:
                    add_step("🧠 Memory:", "Found past research ✅", "step-memory")
                progress_bar.progress(20)

            elif node_name == "orchestrator":
                plan = node_state.get("task_plan", "")[:120]
                add_step("🎯 Orchestrator:", f"Plan ready — {plan}...", "step-orchestrator")
                progress_bar.progress(30)

            elif node_name == "search":
                results = node_state.get("search_results", [])
                arxiv_n = sum(1 for r in results if r.get("source") == "arxiv")
                ss_n = sum(1 for r in results if r.get("source") == "semantic_scholar")
                web_n = sum(1 for r in results if r.get("source") == "web")
                add_step("🔍 Search:",
                         f"ArXiv: {arxiv_n} | Scholar: {ss_n} | Web: {web_n} | Total: {len(results)}",
                         "step-search")
                progress_bar.progress(50)

            elif node_name == "retrieval":
                chunks = node_state.get("rag_context", [])
                add_step("📦 Retrieval:", f"{len(chunks)} chunks from vector DB", "step-retrieval")
                progress_bar.progress(60)

            elif node_name == "synthesis":
                synth = node_state.get("synthesis", "")
                add_step("🧪 Synthesis:", f"Done — {len(synth)} chars synthesized", "step-synthesis")
                progress_bar.progress(75)

            elif node_name == "critic":
                gaps = node_state.get("gaps", [])
                iteration = node_state.get("iteration", 0)
                if gaps:
                    add_step("🔎 Critic:", f"Gaps found → re-searching (iter {iteration})", "step-critic")
                    progress_bar.progress(80)
                else:
                    add_step("✅ Critic:", "Synthesis approved!", "step-critic")
                    progress_bar.progress(88)

            elif node_name == "report":
                report = node_state.get("final_report", "")
                add_step("📝 Report:", "Generated successfully ✅", "step-report")
                add_step("💾 Memory:", "Session saved ✅", "step-memory")
                progress_bar.progress(100)

                # show report
                with right:
                    report_placeholder.empty()
                    st.markdown(report)

                    st.markdown("<br>", unsafe_allow_html=True)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.download_button(
                            label="⬇️ Download Report (.txt)",
                            data=f"Query: {query}\n\n{report}",
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col_b:
                        st.download_button(
                            label="⬇️ Download Report (.md)",
                            data=f"# Research Report\n**Query:** {query}\n\n{report}",
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )

                    # sources expander
                    results = node_state.get("search_results", [])
                    if results:
                        with st.expander(f"📚 View all {len(results)} sources"):
                            for i, r in enumerate(results):
                                title = r.get("title", "N/A")
                                url = r.get("pdf_url") or r.get("url", "")
                                source = r.get("source", "").upper()
                                year = r.get("year") or r.get("published", "")
                                st.markdown(
                                    f"**{i+1}. [{title}]({url})** "
                                    f"`{source}` `{year}`"
                                    if url else
                                    f"**{i+1}. {title}** `{source}` `{year}`"
                                )

        add_step("🎉", "Agent completed successfully!", "step-report")

    except Exception as e:
        add_step("❌ Error:", str(e)[:200], "step-warn")
        with right:
            st.error(f"Agent failed: {e}")

# ── Empty state ──────────────────────────────────────────────────
elif not query:
    st.markdown("""
    <div style='text-align:center; padding: 3rem; color: #718096;'>
        <div style='font-size: 3rem'>🔬</div>
        <h3>Enter a research query above to get started</h3>
        <p>The agent will search ArXiv, Semantic Scholar, and the web,<br>
        then synthesize findings into a structured research report.</p>
    </div>
    """, unsafe_allow_html=True)