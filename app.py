import streamlit as st
import pandas as pd
import joblib
import os
import json
import time
from src.ml.train_model import train_and_save_model
from src.agent.graph import create_agent_graph
from src.rag.vectorstore import get_vectorstore
import dotenv

# Load environment variables
dotenv.load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Zenith AI | Retention Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Header Bar */
    .top-header {
        display: flex;
        align-items: center;
        padding-bottom: 24px;
        margin-bottom: 30px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .top-header h1 { margin: 0; padding: 0; font-weight: 700; color: #ffffff; font-size: 2.2rem; display: flex; align-items: center; gap: 12px; }
    .header-tagline { color: #94a3b8; font-weight: 400; font-size: 1.1rem; margin-top: 8px; }

    /* Glassmorphic Cards */
    .report-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
        color: #f8fafc;
        margin-bottom: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .report-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.7);
    }
    
    /* Risk Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-medium { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-low { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

    /* Tags / Chips for Drivers */
    .driver-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        padding: 6px 14px;
        border-radius: 8px;
        margin: 4px 8px 4px 0;
        font-size: 0.9rem;
        font-weight: 500;
        transition: background 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .driver-tag:hover { background: rgba(56, 189, 248, 0.2); }

    /* Action Item Cards (Recommendations) */
    .action-item {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid #6366f1;
        padding: 20px;
        margin-top: 14px;
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        transition: all 0.3s ease;
    }
    .action-item:hover {
        border-left-color: #818cf8;
        background: rgba(30, 41, 59, 0.8);
        transform: translateX(4px);
    }
    .action-title { font-weight: 600; font-size: 1.15rem; color: #f1f5f9; display:flex; align-items:center; gap: 10px;}
    .action-desc { color: #cbd5e1; font-size: 1rem; line-height: 1.6; }

    /* Stepper / Timeline */
    .timeline-item {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 14px;
        color: #e2e8f0;
        font-size: 1.05rem;
    }
    .timeline-icon {
        color: #6366f1;
        font-size: 1.3rem;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
    }

    /* Target specific streamlit elements for premium feel */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
    }

</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'graph' not in st.session_state:
    st.session_state.graph = None

# --- Setup Operations ---
@st.cache_resource(show_spinner="Loading ML Model & Vector Store...")
def initialize_system():
    # Make sure model exists
    model_path = "src/ml/model.pkl"
    if not os.path.exists(model_path):
        train_and_save_model("data/customer_churn_sample.csv", model_path)
    model = joblib.load(model_path)
    
    # Initialize VectorDB
    get_vectorstore()
    
    # Initialize LangGraph Agent
    graph = create_agent_graph()
    
    return model, graph

try:
    if "GROQ_API_KEY" not in os.environ:
        st.error("🔑 Please set GROQ_API_KEY in your .env file to enable Groq.")
        st.stop()
        
    model, agent_graph = initialize_system()
    st.session_state.model = model
    st.session_state.graph = agent_graph
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

# --- Application UI ---

st.markdown("""
<div class="top-header">
    <div>
        <h1><span style="color:#6366f1;">💠 Zenith</span> AI Retention Intelligence</h1>
        <div class="header-tagline">Advanced Agentic Workflows & Predictive Analytics for Customer Success</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Command Center", "🧠 Architecture & Context", "📊 Model Intelligence"])

with tab2:
    st.header("🧠 Problem Understanding & Business Context")
    st.markdown("""
    **Business Context:**
    Customer churn is a critical metric for subscription and service-based businesses. Retaining an existing customer is significantly cheaper than acquiring a new one. This application aims to not only predict *if* a customer will churn but also provide actionable, agentic AI-driven strategies to prevent it.
    
    **Input-Output Specification:**
    - **Input:** Customer demographic and financial data (Age, Tenure, Balance, etc.)
    - **Output:** Predicted churn risk (0-100%), risk level categorization, key churn drivers, and context-aware retention strategies powered by LLMs and a custom Knowledge Base (RAG).
    
    **System Architecture:**
    1. **ML Model Layer:** Scikit-learn Logistic Regression pipeline predicting churn probability based on structured data.
    2. **Agentic Layer:** Orchestrated using LangGraph for strict State management across operations.
       - *RiskAnalyzer Node:* Interprets ML scores and identifies key drivers (reduced hallucinations via rigid prompting).
       - *Retriever Node:* Queries a ChromaDB vector store for best-practice strategies.
       - *StrategyPlanner Node:* Sythesizes the insights into tailored action plans formatted as strict JSON.
    """)
    
    st.graphviz_chart('''
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor="#4f46e5", fontcolor=white, color=black, fontname="Helvetica"];
        edge [color="#666666"];
        
        Input [label="Customer Data\n(Demographics, Finance)", shape=oval, fillcolor="#2b2b36", fontcolor=white];
        ML [label="ML Predictive Pipeline\n(Scikit-Learn)", fillcolor="#00cc96", fontcolor=black];
        
        subgraph cluster_agent {
            label = "LangGraph Agentic AI Workflow";
            style = dashed;
            color = "#a0a0a0";
            fontcolor=white;
            
            Analyzer [label="Risk Analyzer Agent\n(Groq LLM)"];
            RAG [label="RAG Retriever\n(ChromaDB)"];
            Planner [label="Strategy Planner Agent\n(Strict JSON Schema)"];
            
            Analyzer -> RAG [label=" key risk drivers "];
            RAG -> Planner [label=" retrieved strategies "];
        }
        
        Input -> ML;
        ML -> Analyzer [label=" churn probability "];
        
        Output [label="Structured Retention Report\n(Action Plan + Ethics)", shape=oval, fillcolor="#2b2b36", fontcolor=white];
        Planner -> Output;
    }
    ''')
    
    st.info("Powered by LangGraph (Workflow & State), Chroma RAG, and Groq LLMs.")

with tab3:
    st.header("📊 Model Performance Evaluation Report")
    st.markdown("Below are the evaluation metrics of the underlying Scikit-Learn model on the holdout test set.")
    report_path = "src/ml/evaluation_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report_data = json.load(f)
        
        accuracy = report_data.get("accuracy", 0)
        st.metric("Test Accuracy", f"{accuracy:.2%}")
        
        st.markdown("**Classification Report (Class 1 - Churned):**")
        class_1 = report_data.get("1", {})
        if class_1:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Precision", f"{class_1.get('precision', 0):.2f}")
            col_m2.metric("Recall", f"{class_1.get('recall', 0):.2f}")
            col_m3.metric("F1-Score", f"{class_1.get('f1-score', 0):.2f}")
            
        with st.expander("Show Full Raw Classification Report"):
            st.json(report_data)
    else:
        st.warning("No evaluation report found. Please wait for the model to train.")
        
# --- Sidebar Inputs ---
with st.sidebar:
    st.markdown("### 👤 Profile Configuration")
    st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom: 20px;'>Define customer risk parameters to trigger the AI analysis.</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("**🗣️ Demographics**")
        age = st.slider("Age", 18, 90, 45)
        col1, col2 = st.columns(2)
        with col1:
            geography = st.selectbox("Market", ["France", "Spain", "Germany"])
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])

    with st.container(border=True):
        st.markdown("**💰 Financials & Revenue**")
        tenure = st.number_input("Tenure (Years)", 0, 10, 3)
        balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 100000.0)
        salary = st.number_input("Est. Salary ($)", 10000.0, 250000.0, 80000.0)
        
    with st.container(border=True):
        st.markdown("**📈 Product Engagement**")
        num_products = st.selectbox("Active Products", [1, 2, 3, 4], index=1)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            is_active = st.selectbox("Status", ["Inactive", "Active"])
            is_active = 1 if is_active == "Active" else 0
        with col_c2:
            has_crcard = st.selectbox("Credit Card", ["Yes", "No"])
            has_crcard = 1 if has_crcard == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🚀 Execute AI Risk Analysis")

# --- Main Logic ---
if analyze_button:
    customer_data = {
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "EstimatedSalary": salary,
        "NumOfProducts": num_products,
        "CreditScore": credit_score,
        "Geography": geography,
        "IsActiveMember": is_active,
        "Gender": gender,
        "HasCrCard": has_crcard
    }
    
    with tab1:
        # --- 1. Predictive Intelligence Layer ---
        st.markdown("### 📈 Predictive Intelligence")
        
        df_input = pd.DataFrame([customer_data])
        with st.empty():
            with st.spinner("Initializing predictive engine..."):
                time.sleep(0.6) # Micro interaction feel
                churn_prob = st.session_state.model.predict_proba(df_input)[0][1]

        # Gauge / Progress Simulation
        col_metric, col_assess = st.columns([1, 2])
        with col_metric:
            color_hex = '#ef4444' if churn_prob > 0.5 else ('#f59e0b' if churn_prob > 0.3 else '#10b981')
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.7); padding:24px; border-radius:16px; border:1px solid rgba(255,255,255,0.05); text-align:center; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
                <h4 style="color:#94a3b8; font-weight:500; margin-bottom:10px; font-size:1.05rem; margin-top:0;">Risk Probability Score</h4>
                <div style="font-size:3.5rem; font-weight:700; color:{color_hex}; line-height:1;">
                    {churn_prob:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(churn_prob))
            
        with col_assess:
            st.markdown("<br>", unsafe_allow_html=True)
            if churn_prob > 0.5:
                st.error("⚠️ **CRITICAL RISK DETECTED:** The machine learning model strongly signals an impending churn event. Overriding standard protocol and escalating to Agentic AI workflow for immediate strategic intervention.")
            else:
                st.success("✅ **STABLE PROFILE DETECTED:** Baseline variance detected. Running through Agentic workflow to generate preventative, long-term enrichment strategies.")

        st.markdown("<br><hr style='border-color:rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)
        
        # --- 2. Agentic Reasoning Layer ---
        st.markdown("### 🧠 Autonomous Agent Pipeline")
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom: 20px;'>LangGraph orchestrating Risk Analyzer, Vector DB Retrieval, and Strategy Planning nodes.</p>", unsafe_allow_html=True)
        
        with st.status("Deploying Zenith AI Agents...", expanded=True) as status:
            initial_state = {
                "customer_data": customer_data,
                "churn_score": float(churn_prob),
                "risk_level": "",
                "churn_drivers": [],
                "retrieved_strategies": [],
                "final_recommendations": {},
                "error": ""
            }
            
            st.markdown("<div class='timeline-item'><span class='timeline-icon'>⚡</span> Started execution workflow...</div>", unsafe_allow_html=True)
            
            state = initial_state
            for output in st.session_state.graph.stream(initial_state):
                for node_name, state_update in output.items():
                    time.sleep(0.7) # Provide visual pacing for realism
                    if node_name == "RiskAnalyzer":
                        st.markdown("<div class='timeline-item'><span class='timeline-icon'>🔍</span> <strong>RiskAnalyzer Node:</strong> Successfully isolated key churn drivers across demographics and financials.</div>", unsafe_allow_html=True)
                    elif node_name == "Retriever":
                        st.markdown(f"<div class='timeline-item'><span class='timeline-icon'>📚</span> <strong>RAG Retriever Node:</strong> Extracted {len(state_update.get('retrieved_strategies', []))} contextual protocols from Vector Store.</div>", unsafe_allow_html=True)
                    elif node_name == "StrategyPlanner":
                        st.markdown("<div class='timeline-item'><span class='timeline-icon'>🎯</span> <strong>StrategyPlanner Node:</strong> Synthesized multi-step prescriptive retention blueprint.</div>", unsafe_allow_html=True)
                        
                    state.update(state_update)
                        
            status.update(label="AI Orchestration Complete", state="complete", expanded=False)
            
        if state.get("error"):
            st.error(f"System Fault Detected: {state['error']}")
        else:
            report = state["final_recommendations"]
            risk_profile = report.get("Risk Profile", {})
            risk_level = risk_profile.get("Risk Level", "Unknown")
            
            # --- Output Presentation Layer ---
            st.markdown("<br>### 📋 Prescriptive Retention Blueprint", unsafe_allow_html=True)
            st.markdown("<div class='report-card'>", unsafe_allow_html=True)
            
            # Header Row
            badge_class = "badge-high" if "High" in risk_level else ("badge-medium" if "Medium" in risk_level else "badge-low")
            
            col_h1, col_h2, col_h3 = st.columns(3)
            col_h1.markdown(f"<span class='risk-badge {badge_class}'>Risk Severity: {risk_level}</span>", unsafe_allow_html=True)
            col_h2.markdown(f"<span style='color:#94a3b8; font-size:1.05rem;'>Confidence Matrix:</span> <strong style='color:white; font-size:1.1rem;'>{report.get('Confidence Score', 'Unknown')}</strong>", unsafe_allow_html=True)
            col_h3.markdown(f"<span style='color:#94a3b8; font-size:1.05rem;'>Probability Index:</span> <strong style='color:white; font-size:1.1rem;'>{risk_profile.get('Churn Probability', 'N/A')}</strong>", unsafe_allow_html=True)
            
            st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin: 24px 0;'>", unsafe_allow_html=True)
            
            # Context & Drivers
            st.markdown("<h4 style='color:#e2e8f0; font-size:1.2rem; margin-bottom:14px; display:flex; align-items:center; gap:8px;'>📊 Diagnostic Drivers</h4>", unsafe_allow_html=True)
            drivers_html = "".join([f"<span class='driver-tag'>{driver}</span>" for driver in risk_profile.get("Key Drivers", [])])
            st.markdown(f"<div>{drivers_html}</div>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color:#e2e8f0; font-size:1.2rem; margin-top:28px; margin-bottom:12px; display:flex; align-items:center; gap:8px;'>🧠 Agent Reasoning Context</h4>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:rgba(255,255,255,0.03); padding:16px; border-radius:10px; border-left: 3px solid #64748b; color:#cbd5e1; font-size:1.05rem; line-height: 1.6;'>{report.get('Reasoning', 'No reasoning provided.')}</div>", unsafe_allow_html=True)
            
            # Actionable Steps As Premium Cards
            st.markdown("<h4 style='color:#e2e8f0; font-size:1.2rem; margin-top:32px; margin-bottom:14px; display:flex; align-items:center; gap:8px;'>🎯 Authorized Interventions</h4>", unsafe_allow_html=True)
            actions = report.get("Recommended Actions", [])
            for i, act in enumerate(actions):
                st.markdown(f"""
                <div class='action-item'>
                    <div class='action-title'>
                        <span style="background:rgba(99,102,241,0.2); color:#818cf8; padding:4px 10px; border-radius:6px; font-size:0.85rem; margin-right:12px; font-weight:700;">Task 0{i+1}</span>
                        {act.get('Action', 'Action Item')}
                    </div>
                    <div class='action-desc'>{act.get('Description', '')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Sources integration
            sources = report.get("Sources", [])
            if sources:
                st.markdown("<div style='margin-top:24px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.05)'><p style='color:#64748b; font-size: 0.9rem;'><strong>📖 Grounded Protocol Sources:</strong> " + ", ".join(sources) + "</p></div>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True) # End Report Card
            
            st.markdown(f"<p style='color:#64748b; font-size:0.85rem;'>🔒 <strong>System Disclosure:</strong> {report.get('Disclaimer', 'This strategy is synthesized by AI. Please review thoroughly before executing in the CRM.')}</p>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Show Raw Execution JSON Schema"):
                st.json(report)
