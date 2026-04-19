# PROJECT 5: CUSTOMER CHURN PREDICTION & AGENTIC RETENTION STRATEGY

**From Predictive Analytics to Intelligent Intervention**

## Project Overview
This project involves the design and implementation of an AI-driven customer analytics system that evaluates the probability of a bank customer churning (leaving) and supports data-driven retention decisions. It evolves from a predictive analytical pipeline into a full-fledged agentic AI retention strategist.

In **Milestone 1**, the system applies classical machine learning pipelines to historical bank customer data to predict churn risk, identify key drivers of disengagement, and generate analytical insights.

In **Milestone 2**, the system is extended into an **Agent-Based AI Application (LangGraph & RAG)** that autonomously reasons about that churn risk, retrieves retention best practices, PLANS intervention strategies, and generates structured ethical recommendations.

---

##  Milestone 1: ML-Based Customer Churn Prediction

Financial institutions face significant challenges in retaining customers. Manual identification of at-risk customers is time-consuming, inconsistent, and reactive. This automated customer churn scoring system uses machine learning algorithms to classify customers by churn likelihood.

### Key ML Features
- Upload customer dataset through an interactive UI
- Automatic 6-phase data preprocessing pipeline (missing values, scaling, categorical encoding)
- Training and comparison of multiple ML models simultaneously
- Real-time churn risk prediction for individual customers
- Visualization of evaluation metrics, ROC curves, and feature importance

### Dataset
- **Source:** Maven Analytics — Bank Customer Churn
- **Size:** 10,000 rows × 14 features
- **Target Variable:** Exited (1 = churned, 0 = retained) — ~20% positive class

### Machine Learning Models Evaluated
- **Logistic Regression:** Extracts probabilistic churn classification with `class_weight='balanced'`. 
- **Decision Tree Classifier:** Rule-based classification utilizing `max_depth=10` to identify key risk-driving feature splits.
- **Random Forest Classifier:** Ensemble of 300 decision trees deployed with parallel training (`n_jobs=-1`) to reduce variance.
- **Gradient Boosting Classifier:** Sequential boosting (`learning_rate=0.05`, `subsample=0.8`) achieving consistently highest accuracy on tabular churn data.

---

## Milestone 2 — Agentic Retention Strategy Assistant

### Architecture
The agent uses a 4-node LangGraph StateGraph:
- Risk Profiler: Classifies churn risk tier from ML output
- Strategy Retriever: RAG lookup from ChromaDB retention knowledge base
- Intervention Planner: LLM chain-of-thought reasoning over retrieved context
- Report Generator: Structured JSON retention report generation

### Setup
1. Copy .env.example to .env and add your API key
2. Build the knowledge base: `python rag/build_kb.py`
3. Run: `streamlit run app.py`

### Environment Variables
GEMINI_API_KEY=your_key_here   # Get free at aistudio.google.com

### Ethical Disclosure
This system generates AI-assisted recommendations. All outputs must be 
reviewed by a qualified relationship manager before customer communication. 
Predictions are probabilistic and not guarantees of customer behavior.

##  Team Contribution

| Member | Contribution |
| :--- | :--- |
| **Harsha Gonela (2401010181)** | Model Development, Agent Orchestration (LangGraph), RAG Integration, Streamlit Deployment |
| **Shivansh Upadhyaya (2401020109)** | Data Knowledge Base Construction, Feature Engineering & EDA |
| **Gourav Tanwar (2401010173)** | ML Model Development, Scikit-Learn Pipeline Integration, UI Validation |

---

##  Ethical AI Disclosure
This framework utilizes both deterministic ML thresholds and generative artificial intelligence frameworks to plan user actions. All generated business metrics, reasoning profiles, and action vectors represent automated programmatic insights. Proper manual Quality Assurance (QA) audits must be conducted prior to enacting any financial retention maneuvers.
