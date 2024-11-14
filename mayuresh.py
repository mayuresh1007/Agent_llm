import sys
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# print("jai ganesh")


'''RAG with multi data source'''

# ==================== Wikipedia retriever tool=======================

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper=WikipediaAPIWrapper (top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = WebBaseLoader('https://python.langchain.com/docs/tutorials/rag/')
# loader = WebBaseLoader('https://docs.smith.langchain.com/')
docs = loader.load()
# print(docs)
# sys.exit()
embeddings = OpenAIEmbeddings()
text_spitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs_spitted = text_spitter.split_documents(docs)

vectorDb= FAISS.from_documents(documents=docs_spitted,embedding=embeddings)
retriever = vectorDb.as_retriever()
# print('\nretriever-->\n',retriever)

# ==================== self retriever tool=======================
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(retriever,"langsmith_search","Search fro information about langsmith")
# print('retriever_tool\n',retriever_tool.name)

# ==================== self retriever tool=======================

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper=ArxivAPIWrapper (top_k_results=1, doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
print(arxiv.name)

tools=[wiki,arxiv,retriever_tool]
# print('tools\n',tools)

# ============= Final response ================= 
from langchain_openai import ChatOpenAI
llm= ChatOpenAI(model="gpt-4o-mini",temperature=0.4)


'''Prompt template getting from hub instead write our own'''
from langchain import hub

'''get the prompt from langchain hub
https://python.langchain.com/api_reference/langchain/agents/langchain.agents.openai_functions_agent.base.create_openai_functions_agent.html
'''
prompt=hub.pull("hwchase17/openai-functions-agent")
message = prompt.messages
# print("message\n",message)
# prompt = hub.pull()

'''https://python.langchain.com/api_reference/langchain/agents/langchain.agents.openai_tools.base.create_openai_tools_agent.html'''
from langchain.agents.openai_tools.base import create_openai_tools_agent
# from langchain_community.agent_toolkits import create_openai_tools_agent
# print('create_openapi_tools_agent',create_openai_tools_agent)
# print('create_openapi_agent',create_openapi_agent)
agent = create_openai_tools_agent(llm,tools,prompt)
# print(agent)

## To run Agent We need Agent Executer (variable) tools: list
from langchain.agents import AgentExecutor
agent_executer=AgentExecutor(agent=agent, tools=tools, verbose=True)

# print(agent_executer)

# agent_executer.invoke({"input":"what is langsmith?"})
agent_executer.invoke({"input":"give me how to do serach based on retrieved data based on user query in the form of documenst from vector db?"})


'''python
import numpy as np
from your_vector_db import VectorDB  # Hypothetical vector database library
from your_embedding_model import get_vector  # Hypothetical model to get vector representation

# Step 1: Initialize your vector database
db = VectorDB()

# Step 2: User query
user_query = "What are the benefits of vector databases?"

# Step 3: Convert the user query into a vector
query_vector = get_vector(user_query)

# Step 4: Retrieve similar documents
similar_docs = db.search(query_vector, top_k=10)  # Retrieve top 10 similar documents

# Step 5: Rank and return results
results = sorted(similar_docs, key=lambda x: x['similarity_score'], reverse=True)

for doc in results:
    print(f"Title: {doc['title']}, Similarity Score: {doc['similarity_score']}"
'''