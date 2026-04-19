# PROJECT 5: CUSTOMER CHURN PREDICTION & AGENTIC RETENTION STRATEGY

**From Predictive Analytics to Intelligent Intervention**

## Project Overview
This project involves the design and implementation of an AI-driven customer analytics system that evaluates the probability of a bank customer churning (leaving) and supports data-driven retention decisions. It evolves from a predictive analytical pipeline into a full-fledged agentic AI retention strategist.

In **Milestone 1**, the system applies classical machine learning pipelines to historical bank customer data to predict churn risk, identify key drivers of disengagement, and generate analytical insights.

In **Milestone 2**, the system is extended into an **Agent-Based AI Application (LangGraph & RAG)** that autonomously reasons about that churn risk, retrieves retention best practices, PLANS intervention strategies, and generates structured ethical recommendations.

[**Live Demo - Click here to open the live app**](#)

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

### Evaluation Metrics Calculated
- Accuracy Score
- Precision, Recall, and F1-Score
- Cross-Validated F1 (5-fold)
- ROC-AUC Score & ROC Curve Visualization
- Confusion Matrix

---

##  Milestone 2: Agentic AI Retention Strategy Assistant

Milestone 2 expands the platform by utilizing **LangGraph** (Orchestration) and **ChromaDB** (RAG) coupled with **Groq/Llama-3 LLMs** to dynamically compute retention strategies to save at-risk customers.

### Key Agentic Features & Workflows
1. **State Management (LangGraph):** Explicit state dicts (`TypedDict`) are passed linearly across different reasoning nodes to maintain absolute context.
2. **Risk Analyzer Agent:** Ingests the churn probability score and raw customer features from the ML pipeline to deduce exact behavioral churn drivers natively.
3. **Knowledge Retrieval (RAG via Chroma):** Searches a localized vector database of best-practice retention handbooks utilizing `all-MiniLM-L6-v2` `sentence-transformers`.
4. **Intelligent Strategy Planner:** An LLM planner synthesizes the retrieved knowledge and profile data to generate a multi-step intervention plan structurally constrained via JSON schemas to eliminate hallucinations.
5. **Ethical Disclosures & Sourcing:** All actions are automatically cited with business sources and ethical AI disclaimers integrated deeply into the UI architecture.

---

##  Installation & Setup Instructions

**Step 1: Clone the Repository**
```bash
git clone https://github.com/HARSHA8881/Customer_Churn_Strategy_Assistant.git
cd Customer_Churn_Strategy_Assistant
```

**Step 2: Create a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Configure API Secret**
Create a `.env` file in the root directory and add your LLM API Key:
```env
GROQ_API_KEY="your-groq-api-key"
```

**Step 5: Run the Streamlit Application**
```bash
streamlit run app.py
```
*(The app will initialize the local embedding models, process the vector store, load the ML weights, and deploy the interface to `localhost:8501`)*

---

##  Team Contribution

| Member | Contribution |
| :--- | :--- |
| **Harsha Gonela (2401010181)** | Model Development, Agent Orchestration (LangGraph), RAG Integration, Streamlit Deployment |
| **Shivansh Upadhyaya (2401020109)** |ML Model Development, Scikit-Learn Pipeline Integration, UI Validation |
| **Gourav Tanwar (2401010173)** |Data Knowledge Base Construction, Feature Engineering & EDA |

---

##  Ethical AI Disclosure
This framework utilizes both deterministic ML thresholds and generative artificial intelligence frameworks to plan user actions. All generated business metrics, reasoning profiles, and action vectors represent automated programmatic insights. Proper manual Quality Assurance (QA) audits must be conducted prior to enacting any financial retention maneuvers.
