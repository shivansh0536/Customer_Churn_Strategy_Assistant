import json
from langchain_groq import ChatGroq
from src.agent.state import AgentState
from src.agent.prompts import RISK_ANALYSIS_PROMPT, STRATEGY_PLANNING_PROMPT
from src.rag.vectorstore import retrieve_strategies

def get_llm():
    # Make sure GROQ_API_KEY is in the env
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def risk_analyzer_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = RISK_ANALYSIS_PROMPT.format(
        customer_data=json.dumps(state["customer_data"], indent=2),
        churn_score=f"{state['churn_score']:.2f}"
    )
    
    response = llm.invoke(prompt)
    raw_text = clean_json_response(response.content)
    
    try:
        parsed_data = json.loads(raw_text)
        state["risk_level"] = parsed_data.get("risk_level", "Unknown")
        state["churn_drivers"] = parsed_data.get("churn_drivers", [])
    except json.JSONDecodeError:
        state["risk_level"] = "Error"
        state["churn_drivers"] = ["Failed to parse drivers"]
        state["error"] = "JSON parse error in risk_analyzer"
        
    return state

def retriever_node(state: AgentState) -> AgentState:
    drivers = " ".join(state["churn_drivers"])
    if not drivers:
        drivers = "general retention strategies"
        
    strategies = retrieve_strategies(drivers, k=3)
    state["retrieved_strategies"] = strategies
    return state

def strategy_planner_node(state: AgentState) -> AgentState:
    llm = get_llm()
    
    strategies_text = "\n\n".join(state["retrieved_strategies"])
    
    prompt = STRATEGY_PLANNING_PROMPT.format(
        customer_data=json.dumps(state["customer_data"], indent=2),
        risk_level=state["risk_level"],
        churn_score=f"{state['churn_score']:.2f}",
        churn_drivers=json.dumps(state["churn_drivers"]),
        retrieved_strategies=strategies_text
    )
    
    response = llm.invoke(prompt)
    raw_text = clean_json_response(response.content)
    
    try:
        parsed_data = json.loads(raw_text)
        state["final_recommendations"] = parsed_data
    except json.JSONDecodeError:
        state["error"] = "JSON parse error in strategy_planner"
        state["final_recommendations"] = {
            "Risk Profile": {
                "Risk Level": state["risk_level"],
                "Churn Probability": str(state["churn_score"]),
                "Key Drivers": state["churn_drivers"]
            },
            "Recommended Actions": [{"Action": "Manual Intervention Needed", "Description": "LLM failed to output valid JSON"}],
            "Reasoning": "Parsing Error",
            "Confidence Score": "Low"
        }
        
    return state

def response_generator_node(state: AgentState) -> AgentState:
    # Optional node if we need further formatting. 
    # Current strategy_planner_node already outputs the final format.
    # Just return as is, acts as a pass-through end node.
    return state
