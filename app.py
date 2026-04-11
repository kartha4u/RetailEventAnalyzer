import streamlit as st
import pandas as pd
import boto3
from langchain_aws import ChatBedrock
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Page Config
st.set_page_config(page_title="Retail Data Assistant")
st.title("🛍️ Retail Event Analyzer")

# 2. Load Data
# Assuming your data is in a CSV named 'events.csv'
@st.cache_data
def load_all_data():
    df1 = pd.read_csv("retail_events.csv")
    df2 = pd.read_csv("retail_products.csv")
    df3 = pd.read_csv("retail_customers.csv")
    return df1, df2, df3

df1, df2, df3 = load_all_data()

# 3. Initialize Bedrock LLM
# Make sure your 'region_name' matches where you have model access
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="us.meta.llama4-maverick-17b-instruct-v1:0", # Or your preferred model
    model_kwargs={"temperature": 0}
)
llm.provider_stop_sequence_key_name_map = {"meta": ""}


# 4. Create the Agent

custom_prefix = """
You are a retail data expert working with df1, df2, and df3.
You have access to three dataframes:
- df1: events (Columns event_id, customer_id, product_id, event_type, timestamp. link other dataframe via customer_id/product_id)
- df2: products (Columns product_id, category, brand, price)
- df3: customers (Columns customer_id, age, region, loyalty_tier)

STRICT FORMATTING RULES:
1. You must provide EITHER an 'Action' OR a 'Final Answer'. NEVER both in one turn.
2. After you receive an 'Observation' with the data, stop and provide your 'Final Answer'.
3. After receive Final answer stop iteration and show response.
4. Do NOT include 'Question:' or 'Thought:' multiple times in a single response.

CRITICAL: Do not include any text, links, or markdown code blocks (```) after the 'Final Answer:' line.
"""

agent = create_pandas_dataframe_agent(
    llm,
    [df1, df2, df3],
    verbose=True,
    prefix=custom_prefix,
    allow_dangerous_code=True,
    # This specific string helps the agent self-correct when it fails to parse
    handle_parsing_errors="Check your output format. Ensure 'Final Answer:' is at the very end and not followed by additional text.",
    agent_executor_kwargs={"handle_parsing_errors": True},
    include_df_in_prompt=False,
    agent_type="zero-shot-react-description"  
)

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about the retail events..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = agent.run(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})