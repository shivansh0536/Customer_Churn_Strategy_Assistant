RISK_ANALYSIS_PROMPT = """You are an expert Data Scientist and Customer Success Analyst.
Given a customer's profile and their predicted churn probability, your goal is to analyze the primary reasons they might be at risk of churning.

Customer Data:
{customer_data}

Predicted Churn Probability: {churn_score}

Based on this data, identify:
1. The overall risk level (High, Medium, Low). E.g. > 0.6 is High, 0.4-0.6 is Medium, < 0.4 is Low.
2. The top 2 or 3 specific reasons (drivers) why this customer is likely to churn based on the provided stats.

Output exactly and only a valid JSON object in this format:
{{
  "risk_level": "High/Medium/Low",
  "churn_drivers": ["reason 1", "reason 2"]
}}
"""

STRATEGY_PLANNING_PROMPT = """You are a senior Strategy Planner.
You have the following customer profile, their churn risk profile, and a set of retrieved retention strategies that match their profile.

Customer Data:
{customer_data}

Risk Level: {risk_level}
Churn Probability: {churn_score}
Churn Drivers: {churn_drivers}

Retrieved Strategies from Knowledge Base:
{retrieved_strategies}

Your job is to synthesize these retrieved strategies into actionable recommendations specifically tailored to this customer.
Ensure you structure the final response exactly in the following mandatory JSON format:

{{
  "Risk Profile": {{
    "Risk Level": "{risk_level}",
    "Churn Probability": "{churn_score}",
    "Key Drivers": {churn_drivers}
  }},
  "Recommended Actions": [
    {{
      "Action": "Title of the action",
      "Description": "Detailed explanation of what to do"
    }}
  ],
  "Reasoning": "Explain WHY these recommendations are given based on the customer data.",
  "Confidence Score": "High/Medium/Low",
  "Sources": ["List 2-3 specific best practice methods identified from the retrieved strategies as the sources of your recommendations"],
  "Disclaimer": "This is an AI-generated strategy recommendation. All business and ethical disclosures apply. Review before deploying."
}}

Note: Only provide the valid JSON. No markdown backticks mapping the JSON, just the raw JSON text. Make sure to interpolate the required fields.
"""
