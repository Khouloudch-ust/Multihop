import streamlit as st
import os
import time
import qdrant_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from llama_index.core import Settings

# Set API Keys
os.environ["OPENAI_API_KEY"] = "sk-sl-stPZrlUXvDAgRp1beofE7Jbk5GsbGL9OcbjHT5cT3BlbkFJYrzjtODllOkcnWRPNgzYpqnE2KhL-aIeE4kgwJZUUA"
os.environ["QDRANT_API_KEY"] = "R084qA_ksblgR_oMAebtKKnqGkZRfZxldfiCH6TjojoWbck9Tv4TyQ"

# Initialize clients and models
client = qdrant_client.QdrantClient(
    url="https://225f6124-ed28-4252-a49a-040e264c4d28.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="R084qA_ksblgR_oMAebtKKnqGkZRfZxldfiCH6TjojoWbck9Tv4TyQ",
)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
llm = OpenAI(model="gpt-4")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024
Settings.chunk_overlap = 800



reranker = FlagEmbeddingReranker(
    top_n=10,
    model="BAAI/bge-reranker-large",
)


# Global variable to store the Qdrant index after the documents are uploaded
qdrant_query_index = None
documents_loaded = False
# Load documents and set up VectorStoreIndex
# Function to load documents from user-uploaded files
def load_uploaded_documents(uploaded_files):
    for uploaded_file in uploaded_files:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    documents = SimpleDirectoryReader("data").load_data()
    return documents

# Setup Qdrant VectorStore and Index
def setup_qdrant_index(documents):
    vector_store = QdrantVectorStore(client=client, collection_name="healthcare-insurance")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
    qdrant_query_index=index.as_query_engine(similarity_top_k=20,PostprocessorComponent=[reranker])
    return qdrant_query_index
# Decompose query into subqueries using LangChain

def get_subquery_response_pairs(qdrant_query_index,query,subqueries):
    """
    Function that takes an original query, decomposes it into subqueries,
    queries a basic query engine for responses, and returns a JSON
    containing subquery/response pairs.

    :param query: str : The original query from the user
    :return: dict : JSON object with subquery/response pairs
    """

    # Decompose the original query into subqueries


    # Initialize an empty dictionary to store indexed subquery-response pairs
    subquery_response_pairs = {
        "original_query": query
    }

    # Loop through the subqueries and fetch responses, adding indexed key-value pairs
    for idx, subquery in enumerate(subqueries["sub_queries"], start=1):
        # Get the response from the raw query engine for each subquery
        response = qdrant_query_index.query(subquery)

        # Add subquery and response pair to the dictionary in the desired format
        subquery_response_pairs[f"subquery_{idx}"] = subquery
        subquery_response_pairs[f"response_{idx}"] = response

    # Return the final JSON with indexed subquery/response pairs
    return subquery_response_pairs

# Example usage
# query = "give all copay details for medicare access plus plan and Anthem full dual advantage PPO D-SNP plan"
# result = get_subquery_response_pairs(query)
# print(result)
# Define the SubQuery class as per LangChain's documentation
class SubQuery(BaseModel):
    """Represents a specific sub-query extracted from an original query."""
    sub_query: str = Field(..., description="A specific sub-query to answer the original query.")

# Function to perform query decomposition and return sub-queries as JSON
def decompose_query(original_query: str):
    # Define the system message for decomposition
    # system = """You are an expert at converting user questions into specific sub-queries adapted for vercorstore retrieval.
    # Given a user question, break it down into distinct sub-queries that will help extract relevant context to answer the original question.
    # Each sub-query should focus on a single concept, idea, fact or entity.
    # If there are acronyms or words you are not familiar with, do not try to rephrase them."""
    system = """You have access to documents about healthcare insurance plans separately.
    You are an expert at converting user questions into database queries. \

    Perform query decomposition. Given a user question, break it down into distinct sub questions that \
    you need to answer in order to answer the original question.

    If there are acronyms or words you are not familiar with, do not try to rephrase them.
    example :
    original query : give all deductibles for every plan mentioned in the data.
    sub-query 1 : what are the plans mentioned in the documents?
    sub-query 2 : what are the deductibles for each of the plans mentioned in the documents?
    """

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    #llm = Groq(model="llama3-groq-70b-8192-tool-use-preview")
    #llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview")
    # Bind the SubQuery tool and parser
    llm_with_tools = llm.bind_tools([SubQuery])
    parser = PydanticToolsParser(tools=[SubQuery])

    # Create the query analyzer
    query_analyzer = prompt | llm_with_tools | parser

    # Run the query analyzer
    sub_queries = query_analyzer.invoke({"question": original_query})

    # Convert the result to JSON format
    sub_queries_json = {
        "original_query": original_query,
        "sub_queries": [sub_query.sub_query for sub_query in sub_queries]
    }

    return sub_queries_json



# Fetch subquery responses


# Define the SubQuery class as per LangChain's documentation
class SubQuery(BaseModel):
    """Represents a specific sub-query extracted from an original query."""
    sub_query: str = Field(..., description="A specific sub-query to answer the original query.")

# Function to perform query decomposition and return sub-queries as JSON
def update_subqueries(original_query, subqueries, llm_answer):

    # Define the system message for decomposition
    system = """You have access to documents about healthcare insurance plans separately.
    Given a user question, and a list of sub-queries in the form of steps, update the list of subqueries based on the context provided so that it will help extract relevant context to answer the original complex question.
    If there are acronyms or words you are not familiar with, do not try to rephrase them.
    Always return an updated list of sub-queries.
    """

    # Create the prompt template with correctly referenced variables
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"Original Query: {original_query}\n\n Old list of sub-queries: {subqueries['sub_queries']}\n\nSub-query 1 (treated): {subqueries['sub_queries'][0]}\n\nContext for Sub-query 1: {llm_answer}\n\nUpdate the old list of sub-queries based on the new context provided about subquery_1.\n\nsub_querie : ")
        ]
    )

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Bind the SubQuery tool and parser
    llm_with_tools = llm.bind_tools([SubQuery])
    #print( llm_with_tools)
    parser = PydanticToolsParser(tools=[SubQuery])

    # Create the query analyzer
    query_analyzer = prompt | llm_with_tools | parser

    # Run the query analyzer
    sub_queries = query_analyzer.invoke({
        "original_query": original_query,
        "subqueries": subqueries,
        "subqueries_0": subqueries['sub_queries'][0],
        "llm_answer": llm_answer
    })

    # Convert the result to JSON format
    sub_queries_json = {
        "original_query": original_query,

        "sub_queries": [sub_query.sub_query for sub_query in sub_queries]
    }
    #print(prompt)
    return sub_queries_json


# Generate final answer using the context
def generate_final_answer(step1_result,subquery_response_pairs):
    """
    Function to generate a final answer using an LLM based on the original query, subqueries,
    and their retrieved responses.

    :param subquery_response_pairs: dict : The dictionary containing the original query, subqueries, and responses
    :return: str : The final answer generated by the LLM
    """

    # Extract original query
    original_query = subquery_response_pairs["original_query"]

    # Construct context from subqueries and responses
    context = ""
    context += f"step 1 result: {step1_result}\n\n"
    for idx in range(1, len(subquery_response_pairs) // 2 + 1):  # Dividing by 2 because we have subquery/response pairs
        subquery = subquery_response_pairs[f"subquery_{idx}"]
        response = subquery_response_pairs[f"response_{idx}"]
        context += f"Subquery {idx}: {subquery}\nRetrieved Context {idx}: {response}\n\n"

    # Define the system prompt
    system_message = """You are an expert in answering complex questions by decomposing them into subqueries and using retrieved information to construct a final answer.
    Given the original query, the subqueries, and their retrieved contexts, please provide a comprehensive answer to the original question.
    """

    # Create the chat template for the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", f"Original Query: {original_query}\n\n{context}\nAnswer the original question using the context provided.")
        ]
    )

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0,seed=0)

    # Run the LLM to generate the final answer
    chain = prompt | llm
    final_answer=chain.invoke(    {
        "system_message": system_message,
        "original_query": original_query,
        "context": context,
    })



    # Return the final answer
    return final_answer
def get_unique_citations_and_text(subquery_response_pairs):

    unique_citations = set()  # Use a set to store unique file-page pairs
    text_chunks = []

    for key, response in subquery_response_pairs.items():
        if hasattr(response, 'source_nodes'):  # Checking if response has source_nodes
            source_nodes = response.source_nodes
            for node_with_score in source_nodes:
                node = node_with_score.node
                # Extract metadata, providing fallbacks if not present
                file_name = node.metadata.get('file_name', 'Unknown File')
                page_label = node.metadata.get('page_label', 'Unknown Page')

                # Create a unique identifier for each file-page pair
                citation_key = (file_name, page_label)

                # Add only unique citations
                if citation_key not in unique_citations and (file_name != 'Unknown File' or page_label != 'Unknown Page'):
                    unique_citations.add(citation_key)
                    text_chunks.append(f"Text Chunk (Page {page_label}): {node.text[:200]}...")  # Limit text to first 200 characters

    # Convert set to list for easier formatting
    citations = [f"{file} (Page {page})" for file, page in unique_citations]
    return citations, text_chunks

def pretty_print_llm_output(llm_output):
    """
    Pretty printer function for LLM output, formatting the content, additional info, and metadata.

    :param llm_output: AIMessage : An instance of AIMessage containing LLM output content and metadata
    :return: str : Formatted string for pretty-printed output
    """

    # Extract content from the AIMessage object directly
    content = llm_output.content if hasattr(llm_output, 'content') else 'No content available'
    # additional_kwargs = llm_output.additional_kwargs if hasattr(llm_output, 'additional_kwargs') else {}
    # response_metadata = llm_output.response_metadata if hasattr(llm_output, 'response_metadata') else {}
    # usage_metadata = llm_output.usage_metadata if hasattr(llm_output, 'usage_metadata') else {}

    # Pretty print the output content
    pretty_output = "\n\n"
    pretty_output += content + "\n\n"
    pretty_output += "\n\n"


    return pretty_output


# Streamlit layout and interactions
st.title("Healthcare Plan Query Assistant")
# Streamlit layout and interactions
st.title("Healthcare Plan Query Assistant")

# Check if documents and index exist in session state
if "qdrant_query_index" not in st.session_state:
    st.session_state.qdrant_query_index = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Upload section
uploaded_files = st.file_uploader("Upload healthcare plan documents", type=["pdf"], accept_multiple_files=True)

# Process the documents and create an index only once after documents are uploaded
if uploaded_files and not st.session_state.documents_loaded:
    # Load documents
    with st.spinner('Loading documents...'):
        documents = load_uploaded_documents(uploaded_files)

    # Setup Qdrant Index
    with st.spinner('Setting up Qdrant index...'):
        st.session_state.qdrant_query_index = setup_qdrant_index(documents)
        st.session_state.documents_loaded = True
        st.success("Indexing complete. You can now ask questions!")

# Enter the query
query = st.text_input("Enter your query:")

if st.button("Submit") and query and st.session_state.qdrant_query_index:
    # Time the query response
    start_time = time.time()

    # Decompose query
    with st.spinner('Decomposing query...'):
        subqueries = decompose_query(query)
        step1_result = st.session_state.qdrant_query_index.query(subqueries['sub_queries'][0])

    # Get subquery responses and generate final answer
    with st.spinner('Fetching responses and generating final answer...'):
        updated_subqueries = update_subqueries(query, subqueries, step1_result)
        subquery_response_pairs = get_subquery_response_pairs(st.session_state.qdrant_query_index, query, updated_subqueries)
        answer = generate_final_answer(step1_result, subquery_response_pairs)
        final_answer=pretty_print_llm_output(answer)


    # Display the final answer
    st.subheader("Final Answer:")
    st.write(final_answer)

    # Display citations and text chunks
    st.subheader("Citations and Relevant Text Chunks:")
    citations, text_chunks = get_unique_citations_and_text(subquery_response_pairs)
    st.write("Citations:")
    for citation in citations:
        st.write(citation)
    # st.write("Relevant Text Chunks:")
    # for chunk in text_chunks:
    #     st.write(chunk)
    #Display response time
    st.write(f"Response returned in {time.time() - start_time:.2f} seconds.")