import os
import streamlit as st
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.embeddings import Embeddings

# 1. Dynamic-quantize the SentenceTransformer model
@st.cache_resource(ttl=3600)
def load_quantized_model():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return qmodel

# 2. Wrap quantized model for LangChain
class QuantizedEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# 3. Keep-alive ping endpoint
if "ping" in st.query_params:
    st.write("pong")
    st.stop()

# 4. Load GROQ API key
load_dotenv()
#st.secrets.get("GROQ_API_KEY") or
GROQ_API_KEY =  os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing API keys!")
    st.stop()

st.title("Conversational RAG With PDF Uploads and Chat History")

# 5. Cache PDF loading
@st.cache_data(max_entries=3, persist="disk")
def load_and_process_pdfs(paths):
    docs = []
    for p in paths:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())
    return docs

# 6. Cache FAISS index creation
@st.cache_resource(ttl=3600)
def generate_vectorstore(_docs):
    splits = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500).split_documents(_docs)
    embeddings = QuantizedEmbeddings(load_quantized_model())
    return FAISS.from_documents(splits, embeddings)

# 7. Cache LLM client
@st.cache_resource(ttl=3600)
def load_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")

# Initialize sources
documents = load_and_process_pdfs(["Health Montoring Box (CHATBOT).pdf"])
vectorstore = generate_vectorstore(documents)
retriever = vectorstore.as_retriever()
llm = load_llm()

# Build RAG chain (unchanged)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history + latest question, reformulate it into a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are a medical assistant for question-answering tasks. "
    "Use retrieved context to answer concisely. "
    "If values are abnormal, advise consulting a doctor. "
    "If unknown, say so. Keep answers to 1–2 sentences.\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_session_history(sid: str) -> BaseChatMessageHistory:
    if sid not in st.session_state:
        st.session_state[sid] = ChatMessageHistory()
    return st.session_state[sid]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# UI: Session ID & question input
session_id = st.text_input("Session ID", "default_session")
user_input = st.text_input("Your question:", key="user_input")
if st.button("Submit"):
    history = get_session_history(session_id)
    with st.spinner("Generating response..."):
        output = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        answer = output["answer"].strip()
        st.markdown(f"**Answer:** {answer}")
        with st.expander("Chat History"):
            st.write(history.messages)

# Clear history
if st.button("Clear Chat History"):
    st.session_state[session_id] = ChatMessageHistory()
    st.success("Chat history cleared!")
