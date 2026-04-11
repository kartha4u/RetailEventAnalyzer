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
You are working with three pandas DataFrames:

- df_0 → retail_events data(event_id, customer_id, product_id, event_type, timestamp.)
- df_1 → retail_products data (product_id, category, brand, price.)
- df_2 → retail_customers data (customer_id, age, region, loyalty_tier.)

Rules:
- Always use Python pandas code
- Use df_0, df_1, df_2 exactly as named
- Join DataFrames when needed:
    df_0.product_id = df_1.product_id
    df_0.customer_id = df_2.customer_id
- Return clear final answers.
- Do not mention dataframes, df_1, df_2, df_3 in responses. 
"""

agent = create_pandas_dataframe_agent(
    llm,
    [df_0, df_1, df_2],
    verbose=True,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
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

