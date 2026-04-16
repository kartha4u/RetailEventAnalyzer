import streamlit as st
import pandas as pd
import boto3
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.vectorstores import FAISS
from langchain.tools import tool

# 1. Page Config
st.set_page_config(page_title="Personalized Recommendation System")
st.title("🎯 Personalized Recommendation System")

# 2. Load Data
@st.cache_data
def load_all_data():
    df1 = pd.read_csv("retail_events.csv")
    df2 = pd.read_csv("retail_products.csv")
    df3 = pd.read_csv("retail_customers.csv")
    return df1, df2, df3

df1, df2, df3 = load_all_data()

# 3. Initialize Bedrock
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="amazon.nova-pro-v1:0", 
    model_kwargs={"temperature": 0}
) 

embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v2:0"
)

# 4. Data Preparation - Embeddings Vector Store
@st.cache_resource
def prepare_vector_store(_df_products):
    texts = []
    metadatas = []
    for _, row in _df_products.iterrows():
        text = f"Product {row['product_id']} is a {row['category']} item from brand {row['brand']} priced at ${row['price']}."
        texts.append(text)
        metadatas.append(row.to_dict())
    
    if texts:
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        return vectorstore
    return None

with st.spinner("Initializing Recommendation Engine..."):
    vectorstore = prepare_vector_store(df2)

@tool
def semantic_product_search(query: str) -> str:
    """Use this tool to find real-time product recommendations based on a natural language query describing the desired product category, brand, or style. This searches the embedded vector database."""
    if vectorstore is None:
        return "Vector store not initialized."
    docs = vectorstore.similarity_search(query, k=5)
    
    results = []
    for d in docs:
        results.append(d.page_content)
    
    return "Similar products found:\n" + "\n".join(results)


# 5. Create the Agent
custom_prefix = """
You are a Personalized Recommendation System expert. 
You have access to three dataframes:
- df1: events (past user behavior: view, cart, purchase - use to evaluate what the user likes)
- df2: products (product catalog)
- df3: customers (demographics)

You also have a tool `semantic_product_search` which performs a similarity search over all products.

GOAL: Provide real-time product recommendations. 
When asked to recommend products for a customer (e.g. C000021), follow these steps:
1. Examine df3 to see the customer demographic.
2. Examine df1 to see what products the customer has viewed, carted, or purchased in the past.
3. Identify the customer's preferred categories or brands from df2 based on step 2.
4. Formulate a search query (e.g. "Electronics by Acme") based on the customer's preferences and use the `semantic_product_search` tool to get relevant products from the catalog.
5. Provide a personalized recommendation response combining their history and new matching products. Make sure the response is well-formatted and explains WHY you recommended these products.

STRICT FORMATTING RULES:
1. You must provide EITHER an 'Action' OR a 'Final Answer'. NEVER both in one turn.
2. After you receive an 'Observation' with the data, stop and provide your 'Final Answer'.
3. Do NOT include 'Question:' or 'Thought:' multiple times in a single response.
4. CRITICAL: Do not include any text, links, or markdown code blocks (```) after the 'Final Answer:' line.
"""

agent = create_pandas_dataframe_agent(
    llm,
    [df1, df2, df3],
    verbose=True,
    prefix=custom_prefix,
    extra_tools=[semantic_product_search],
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
    include_df_in_prompt=False,
    agent_type="zero-shot-react-description"  
)

# 6. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me for recommendations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing profile and generating recommendations..."):
            try:
                response = agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating recommendation: {e}")