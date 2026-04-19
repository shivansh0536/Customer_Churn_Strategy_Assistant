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
  page_icon="",
  layout="wide",
  initial_sidebar_state="expanded"
)

# --- Premium Custom CSS ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  
  /* Top Hero Banner Match */
  .top-hero {
    background: linear-gradient(135deg, #16243d 0%, #0f4b66 100%);
    border-radius: 14px;
    padding: 40px;
    color: white;
    margin-bottom: 40px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
  }
  .hero-badge {
    background-color: #00d2b4;
    color: #022c22;
    font-weight: 800;
    font-size: 0.7rem;
    padding: 6px 14px;
    border-radius: 20px;
    text-transform: uppercase;
    display: inline-block;
    margin-bottom: 24px;
    letter-spacing: 1px;
  }
  .hero-title { 
    font-size: 2.8rem; 
    font-weight: 800; 
    margin-bottom: 16px; 
    line-height: 1.2;
    color: #ffffff;
  }
  .hero-subtitle { 
    color: #e2e8f0; 
    font-size: 1.05rem; 
    max-width: 800px; 
    line-height: 1.6;
    font-weight: 400;
  }
  
  /* Section Headers */
  .section-header {
    display: flex;
    align-items: center;
    font-size: 1.4rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 30px;
    margin-bottom: 20px;
  }
  .section-header::after {
    content: "";
    flex: 1;
    margin-left: 20px;
    height: 2px;
    background-color: #e2e8f0;
  }

  /* System Dashboard Cards */
  .info-card {
    background-color: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    height: 100%;
    display: flex;
    flex-direction: column;
    border: 1px solid #f1f5f9;
    margin-bottom: 16px;
  }
  .card-title { font-size: 0.75rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; margin-bottom: 15px; letter-spacing: 1px;}
  .card-value { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 15px; line-height: 1; }
  .card-pill {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    width: fit-content;
  }
  
  .pill-green { background-color: #dcfce7; color: #166534; }
  .pill-blue { background-color: #dbeafe; color: #1e40af; }
  .pill-purple { background-color: #f3e8ff; color: #6b21a8; }
  .pill-red { background-color: #fee2e2; color: #991b1b; }
  .pill-gray { background-color: #f1f5f9; color: #475569; }

  /* Number Badge for Quick Start structure */
  .num-badge {
    background-color: #3b82f6;
    color: white;
    width: 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-weight: bold;
    font-size: 0.9rem;
    margin-right: 12px;
  }

  /* Actionable items in Agent output */
  .action-row {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
  }
  .action-desc { color: #475569; line-height: 1.6; font-size: 0.95rem; }

  /* Buttons */
  div.stButton > button, button[data-testid="baseButton-secondary"] {
    width: 100% !important;
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
  div.stButton > button:hover, button[data-testid="baseButton-secondary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
  }
  div.stButton > button *, button[data-testid="baseButton-secondary"] * {
    color: white !important;
  }

  /* Fixed Sidebar Typography overrides */
  [data-testid="stSidebar"] * {
    /* This prevents overriding specific input background colors but helps text */
    border-color: rgba(255,255,255,0.1);
  }
  /* Force widget labels (the text above inputs) to be white */
  [data-testid="stSidebar"] label p {
    color: #f8fafc !important;
  }
  /* Force specific slider numbers (like 18-90) to be white/light */
  [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
    color: #cbd5e1 !important;
  }
  [data-testid="stSidebar"] .stSlider div {
    color: #f8fafc !important;
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
    st.error(" Please set GROQ_API_KEY in your .env file to enable Groq.")
    st.stop()
    
  model, agent_graph = initialize_system()
  st.session_state.model = model
  st.session_state.graph = agent_graph
except Exception as e:
  st.error(f"Error initializing system: {e}")
  st.stop()

# --- Application UI ---

st.markdown("""
<div class="top-hero">
  <div class="hero-badge">Agentic AI • Customer Retention</div>
  <div class="hero-title">Welcome to the Customer Churn Strategy Framework!</div>
  <div class="hero-subtitle">
    This system evaluates the probability of a bank customer leaving. Built on classical ML principles and advanced generative AI (LangGraph + RAG), we've integrated predictive modeling with intelligent, autonomous Agent workflows into a clean, interactive interface. Navigate the sidebar to test predictions and generate retention blueprints.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section-header'>System Dashboard</div>", unsafe_allow_html=True)
col_d1, col_d2, col_d3, col_d4 = st.columns(4)

with col_d1:
  st.markdown("""
  <div class='info-card'>
    <div class='card-title'>Models Trained</div>
    <div class='card-value'>1</div>
    <div class='card-pill pill-green'>Ready to predict</div>
  </div>
  """, unsafe_allow_html=True)
  
with col_d2:
  st.markdown("""
  <div class='info-card'>
    <div class='card-title'>LLM Backend</div>
    <div class='card-value'>Groq</div>
    <div class='card-pill pill-blue'>Llama 3 Connected</div>
  </div>
  """, unsafe_allow_html=True)
  
with col_d3:
  st.markdown("""
  <div class='info-card'>
    <div class='card-title'>Knowledge Base</div>
    <div class='card-value'>Chroma</div>
    <div class='card-pill pill-purple'>Semantic Search active</div>
  </div>
  """, unsafe_allow_html=True)
  
with col_d4:
  st.markdown("""
  <div class='info-card'>
    <div class='card-title'>Framework</div>
    <div class='card-value'>Streamlit</div>
    <div class='card-pill pill-blue'>+ LangGraph</div>
  </div>
  """, unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs([" Command Center", " Architecture Status", " Performance Evaluation"])

with tab2:
  st.header(" Problem Understanding & Business Context")
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
  st.header(" Model Performance Evaluation Report")
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
  st.markdown("<h3 style='color:white; margin-bottom: 20px;'>Customer Profile Config</h1>", unsafe_allow_html=True)
  st.markdown("<p style='color:#cbd5e1; font-size:0.95rem; margin-bottom: 20px;'>Define customer risk parameters to trigger the AI analysis.</p>", unsafe_allow_html=True)
  
  st.markdown("<strong style='color:white'> Demographics</strong>", unsafe_allow_html=True)
  age = st.slider("Age", 18, 90, 45)
  col1, col2 = st.columns(2)
  with col1:
    geography = st.selectbox("Market", ["France", "Spain", "Germany"])
  with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

  st.markdown("<br><strong style='color:white'> Financials & Revenue</strong>", unsafe_allow_html=True)
  tenure = st.number_input("Tenure (Years)", 0, 10, 3)
  balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 100000.0)
  salary = st.number_input("Est. Salary ($)", 10000.0, 250000.0, 80000.0)
  
  st.markdown("<br><strong style='color:white'> Product Engagement</strong>", unsafe_allow_html=True)
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
  analyze_button = st.button(" Evaluate Individual Risk Profile")

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
    st.markdown("<div class='section-header'>Predictive Analysis Output</div>", unsafe_allow_html=True)
    
    df_input = pd.DataFrame([customer_data])
    with st.empty():
      with st.spinner("Initializing predictive engine..."):
        time.sleep(0.4)
        churn_prob = st.session_state.model.predict_proba(df_input)[0][1]

    # White info card style for outputs
    pill_class = 'pill-red' if churn_prob > 0.5 else ('pill-blue' if churn_prob > 0.3 else 'pill-green')
    status_text = 'High Risk' if churn_prob > 0.5 else 'Stable'
    
    st.markdown(f"""
    <div class='info-card'>
      <div class='card-title'>Churn Probability Score (Random Forest)</div>
      <div class='card-value'>{churn_prob:.1%}</div>
      <div class='card-pill {pill_class}'>{status_text}</div>
      <p style='color:#475569; margin-top:15px; font-size:0.95rem;'>
        {' Priority intervention required. The predictive pipeline strongly signals risk.' if churn_prob > 0.5 
        else ' Baseline variance. No immediate predictive risk flags.'}
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- 2. Agentic Reasoning Layer - Quick Start Format ---
    st.markdown("<div class='section-header'>Agentic Intervention Strategy</div>", unsafe_allow_html=True)
    
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
      
      st.markdown("<div class='timeline-item'><span class='timeline-icon'></span> Started execution workflow...</div>", unsafe_allow_html=True)
      
      state = initial_state
      try:
        for output in st.session_state.graph.stream(initial_state):
          for node_name, state_update in output.items():
            time.sleep(0.7) # Provide visual pacing for realism
            if node_name == "RiskAnalyzer":
              st.markdown("<div class='timeline-item'><span class='timeline-icon'></span> <strong>RiskAnalyzer Node:</strong> Successfully isolated key churn drivers across demographics and financials.</div>", unsafe_allow_html=True)
            elif node_name == "Retriever":
              st.markdown(f"<div class='timeline-item'><span class='timeline-icon'></span> <strong>RAG Retriever Node:</strong> Extracted {len(state_update.get('retrieved_strategies', []))} contextual protocols from Vector Store.</div>", unsafe_allow_html=True)
            elif node_name == "StrategyPlanner":
              st.markdown("<div class='timeline-item'><span class='timeline-icon'></span> <strong>StrategyPlanner Node:</strong> Synthesized multi-step prescriptive retention blueprint.</div>", unsafe_allow_html=True)
              
            state.update(state_update)
      except Exception as e:
        status.update(label="API Connectivity Offline or Execution Failed", state="error", expanded=True)
        st.error(f"API Connectivity or processing error occurred. Please try again later. Details: {str(e)}")
        state["error"] = "API Limit Reached or Timeout"
            
      status.update(label="AI Orchestration Complete", state="complete", expanded=False)
      
    if state.get("error"):
      st.error(f"System Fault Detected: {state['error']}")
    else:
      report = state["final_recommendations"]
      risk_profile = report.get("Risk Profile", {})
      
      # --- Output Presentation Layer - Structured like the reference image list ---
      # Diagnostic Data
      drivers_html = "".join([f"<p style='margin: 4px 0;'>• {driver}</p>" for driver in risk_profile.get("Key Drivers", [])])
      
      st.markdown(f"""
      <div class='action-row'>
        <div class='num-badge'>1</div>
        <div>
          <h4 style='margin:0 0 8px 0; color:#0f172a'>Risk Matrix & Key Drivers</h4>
          <div class='action-desc'>
            {drivers_html}
          </div>
        </div>
      </div>
      """, unsafe_allow_html=True)
      
      st.markdown(f"""
      <div class='action-row'>
        <div class='num-badge'>2</div>
        <div>
          <h4 style='margin:0 0 8px 0; color:#0f172a'>Agentic Reasoning Summary</h4>
          <div class='action-desc'>{report.get('Reasoning', '')}</div>
        </div>
      </div>
      """, unsafe_allow_html=True)
      
      # Actionable Steps As Reference Format
      actions = report.get("Recommended Actions", [])
      for i, act in enumerate(actions):
        st.markdown(f"""
        <div class='action-row'>
          <div class='num-badge'>{i+3}</div>
          <div>
            <h4 style='margin:0 0 8px 0; color:#0f172a'>{act.get('Action', 'Prescriptive Action')}</h4>
            <div class='action-desc'>{act.get('Description', '')}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
      
      st.markdown(f"<p style='color:#64748b; font-size:0.85rem; margin-top:20px;'> <strong>System Disclosure:</strong> {report.get('Disclaimer', '')}</p>", unsafe_allow_html=True)
      
      st.markdown("<br>", unsafe_allow_html=True)
      with st.expander("Show Raw Execution JSON Schema"):
        st.json(report)
