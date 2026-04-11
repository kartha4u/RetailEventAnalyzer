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
    df_0 = pd.read_csv("retail_events.csv")
    df_1 = pd.read_csv("retail_products.csv")
    df_2 = pd.read_csv("retail_customers.csv")
    return df_0, df_1, df_2

df_0, df_1, df_2 = load_all_data()

# 3. Initialize Bedrock LLM
# Make sure your 'region_name' matches where you have model access
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="us.meta.llama3-3-70b-instruct-v1:0", # Or your preferred model
    model_kwargs={"temperature": 0}
)
llm.provider_stop_sequence_key_name_map = {"meta": ""}


# 4. Create the Agent

custom_prefix = """
You are a retail data expert working with df1, df2, and df3.
You have access to three dataframes:
- df1: events (link via customer_id/product_id)
- df2: products
- df3: customers

STRICT FORMATTING RULES:
1. You must provide EITHER an 'Action' OR a 'Final Answer'. NEVER both in one turn.
2. After you receive an 'Observation' with the data, stop and provide your 'Final Answer'.
3. Do NOT include 'Question:' or 'Thought:' multiple times in a single response.
4. If the code `df1['customer_id'].nunique()` returns 800, your next response must be:
   Final Answer: There are 800 unique customers.
"""

agent = create_pandas_dataframe_agent(
    llm,
    [df_0, df_1, df_2],
    verbose=True,
    prefix=custom_prefix,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
    include_df_in_prompt=False 
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